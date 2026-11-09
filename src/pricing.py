"""Monte Carlo pricing engine for Synthetic and Autocall structured products.

The SmartDelta pricing screen (Section 1, step 11 "Price" and Section 4
"Details that matter for product design") specifies:

* up to 10 underlying assets;
* a configurable interest-rate term structure (flat or US Treasury curve);
* Monte Carlo simulation with 100 000 paths;
* European barriers only, up to 5 on the maturity date;
* a Synthetic payoff with capital protection and no coupons, and an Autocall
  payoff with memory feature, observation schedule, autocall barrier, coupon
  barrier and coupon rate.

The simulation draws correlated standard-normal variates via a Cholesky
decomposition of the correlation matrix and evolves each asset under the
risk-neutral geometric Brownian motion
    dS_i / S_i = (r(t) - q_i) dt + sigma_i dW_i
where ``q_i`` is the continuous dividend yield (Ctv in the paper) and the
instantaneous short rate ``r(t)`` is taken from a ``YieldCurve``.

The payoff formulas are implemented from scratch because the paper gives no
closed forms (Section 4 marks "Upside participation and leverage" as "Discussed
at the demo, not seen on screen").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .config import RANDOM_SEED, TRADING_DAYS_PER_YEAR
from .rates import YieldCurve


# -----------------------------------------------------------------------------
# Product definitions
# -----------------------------------------------------------------------------


@dataclass
class Barrier:
    """European barrier applied at maturity.

    ``level`` is given as a fraction of the initial basket value. When the
    basket's terminal value is below ``level`` the ``participation`` rate is
    applied to the shortfall (``direction = -1``) or to the excess above the
    strike (``direction = +1``). ``cap`` optionally caps the payoff
    contribution to a fixed fraction of par.
    """

    level: float
    direction: int  # +1 (upside) / -1 (downside protection breach)
    participation: float = 1.0
    cap: float | None = None


@dataclass
class SyntheticProduct:
    """Capital-protected synthetic note on a basket.

    Payoff at maturity (fraction of par):
        protection + participation * max(0, basket_ret) + Sum(barriers)

    where ``basket_ret = sum_i w_i * (S_i(T) / S_i(0)) - 1``. Up to
    ``MAX_MATURITY_BARRIERS`` barriers may be attached at the maturity date
    (Section 4 "Up to 5 barriers on the maturity date").
    """

    par_value: float
    weights: np.ndarray
    maturity_years: float
    protection: float
    participation: float
    barriers: list[Barrier] = field(default_factory=list)
    name: str = "Synthetic"

    def payoff(self, terminal: np.ndarray, spot0: np.ndarray) -> np.ndarray:
        """Vectorized payoff over Monte Carlo paths."""
        basket_ret = (terminal / spot0) @ self.weights - 1.0
        base = self.protection + self.participation * np.maximum(basket_ret, 0.0)
        extra = np.zeros_like(basket_ret)
        for b in self.barriers:
            if b.direction > 0:
                contrib = b.participation * np.maximum(basket_ret + 1.0 - b.level, 0.0)
            else:
                contrib = -b.participation * np.maximum(b.level - (basket_ret + 1.0), 0.0)
            if b.cap is not None:
                contrib = np.clip(contrib, -abs(b.cap), abs(b.cap))
            extra += contrib
        return self.par_value * (base + extra)


@dataclass
class AutocallProduct:
    """Single-underlying autocall note with memory coupons.

    At each observation date ``t_k``:
      * if ``S(t_k) >= autocall_barrier * S(0)`` the note pays par plus all
        accrued coupons and is redeemed early.
      * else if ``S(t_k) >= coupon_barrier * S(0)`` a coupon is paid; if
        memory is on, any missed coupons are paid too.

    At maturity, if not autocalled:
      * if ``S(T) >= protection_barrier * S(0)`` the note pays par (plus the
        last coupon if due and all memory-accrued coupons).
      * else the note pays par * (S(T) / S(0)).

    The paper pins these mechanics (Section 1, step 10 "Structure - Autocall")
    but gives no closed form.
    """

    par_value: float
    maturity_years: float
    obs_freq_months: int
    autocall_barrier: float
    coupon_barrier: float
    protection_barrier: float
    coupon_rate: float
    memory: bool
    name: str = "Autocall"

    def observation_times(self) -> np.ndarray:
        """Observation dates in years, excluding t=0 and ending at maturity."""
        n_obs = int(round(self.maturity_years * 12.0 / self.obs_freq_months))
        return np.array([
            (k + 1) * self.obs_freq_months / 12.0 for k in range(n_obs)
        ], dtype=float)


# -----------------------------------------------------------------------------
# Monte Carlo engine
# -----------------------------------------------------------------------------


@dataclass
class MarketParams:
    """Risk-neutral market inputs for the simulation."""

    spot: np.ndarray            # (n_assets,)
    vol: np.ndarray             # (n_assets,) annualized
    div_yield: np.ndarray       # (n_assets,) continuous
    correlation: np.ndarray     # (n_assets, n_assets)
    tickers: Sequence[str]


def _validate_market(mp: MarketParams) -> None:
    n = len(mp.tickers)
    for arr in (mp.spot, mp.vol, mp.div_yield):
        if arr.shape != (n,):
            raise ValueError("spot / vol / div_yield must all have shape (n_assets,)")
    if mp.correlation.shape != (n, n):
        raise ValueError("correlation must be square (n_assets, n_assets)")
    if not np.allclose(mp.correlation, mp.correlation.T, atol=1e-8):
        raise ValueError("correlation matrix must be symmetric")
    if np.any(np.diag(mp.correlation) <= 0):
        raise ValueError("correlation diagonal must be positive")


def simulate_paths(
    market: MarketParams,
    curve: YieldCurve,
    n_paths: int,
    maturity_years: float,
    obs_times: np.ndarray | None = None,
    steps_per_year: int = TRADING_DAYS_PER_YEAR,
    seed: int = RANDOM_SEED,
    antithetic: bool = True,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate correlated GBM paths under the risk-neutral measure.

    Returns a tuple ``(times, paths)`` where ``times`` is shape ``(n_steps+1,)``
    (always including 0 at the front and ``maturity_years`` at the end) and
    ``paths`` is shape ``(n_paths, n_steps+1, n_assets)``.

    When ``obs_times`` is given, the grid is restricted to those times (plus 0
    and ``maturity_years``) instead of a dense daily step; this keeps memory
    bounded for products that only need a few observation dates.

    Paths are simulated in chunks of ``chunk_size`` to cap peak memory;
    variance reduction via antithetic variates halves the independent draws so
    ``n_paths`` must be even in that case.
    """
    _validate_market(market)
    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2")

    if obs_times is None:
        n_steps = max(1, int(round(maturity_years * steps_per_year)))
        times = np.linspace(0.0, maturity_years, n_steps + 1)
    else:
        grid = np.unique(np.concatenate(([0.0], np.asarray(obs_times, float),
                                         [maturity_years])))
        grid = grid[(grid >= 0.0) & (grid <= maturity_years + 1e-12)]
        times = grid
        n_steps = len(times) - 1

    n_assets = len(market.tickers)
    L = np.linalg.cholesky(market.correlation + 1e-12 * np.eye(n_assets))

    dts = np.diff(times)
    # Piecewise-constant forward rates from the curve across each segment.
    fwd = np.array([curve.forward(times[i], times[i + 1]) for i in range(n_steps)])
    vol = market.vol
    q = market.div_yield
    drift = (fwd[:, None] - q[None, :] - 0.5 * vol[None, :] ** 2) * dts[:, None]
    sqrt_dt = np.sqrt(dts)

    rng = np.random.default_rng(seed)
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic variates are used")

    paths = np.empty((n_paths, n_steps + 1, n_assets), dtype=np.float64)
    paths[:, 0, :] = market.spot[None, :]
    draws_per_pair = n_steps * n_assets

    done = 0
    while done < n_paths:
        if antithetic:
            half = min(chunk_size // 2, (n_paths - done) // 2)
            base = rng.standard_normal(size=(half, n_steps, n_assets))
            z = np.concatenate([base, -base], axis=0)
            del base
        else:
            take = min(chunk_size, n_paths - done)
            z = rng.standard_normal(size=(take, n_steps, n_assets))
        # Correlate across assets: (chunk, steps, assets) @ L.T
        z = z @ L.T
        log_steps = drift[None, :, :] + vol[None, None, :] * sqrt_dt[None, :, None] * z
        del z
        log_cum = np.cumsum(log_steps, axis=1)
        del log_steps
        chunk_paths = market.spot[None, None, :] * np.exp(log_cum)
        del log_cum
        end = done + chunk_paths.shape[0]
        paths[done:end, 1:, :] = chunk_paths
        done = end
        del chunk_paths
        _ = draws_per_pair  # silence unused warning in future edits

    return times, paths


def price_synthetic(
    product: SyntheticProduct,
    market: MarketParams,
    curve: YieldCurve,
    n_paths: int,
    seed: int = RANDOM_SEED,
) -> dict:
    """Price a synthetic capital-protected product by Monte Carlo."""
    if product.weights.shape != (len(market.tickers),):
        raise ValueError("weights and market tickers must have matching length")
    if not np.isclose(product.weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")

    # Synthetic payoff is European (terminal only), so a single-step grid is
    # exact under GBM with piecewise-constant drift and keeps memory bounded.
    times, paths = simulate_paths(
        market=market,
        curve=curve,
        n_paths=n_paths,
        maturity_years=product.maturity_years,
        obs_times=np.array([product.maturity_years]),
        seed=seed,
    )
    terminal = paths[:, -1, :]
    payoff = product.payoff(terminal, market.spot)
    discount = curve.discount(product.maturity_years)
    present_values = discount * payoff
    price = float(present_values.mean())
    mc_stderr = float(present_values.std(ddof=1) / np.sqrt(n_paths))

    # Expected annualized yield based on expected payoff.
    expected_yield = float(np.log(payoff.mean() / product.par_value) / product.maturity_years)

    return {
        "product": product.name,
        "price": price,
        "mc_stderr": mc_stderr,
        "mc_ci95": (price - 1.96 * mc_stderr, price + 1.96 * mc_stderr),
        "expected_payoff": float(payoff.mean()),
        "expected_yield": expected_yield,
        "prob_loss": float(np.mean(payoff < product.par_value)),
        "n_paths": n_paths,
        "discount_factor": discount,
        "terminal_percentiles": {
            "p05": float(np.percentile(payoff, 5)),
            "p50": float(np.percentile(payoff, 50)),
            "p95": float(np.percentile(payoff, 95)),
        },
        "payoff_sample": payoff[:5000],  # for plotting, kept small
    }


def price_autocall(
    product: AutocallProduct,
    market: MarketParams,
    curve: YieldCurve,
    n_paths: int,
    seed: int = RANDOM_SEED,
) -> dict:
    """Price a single-underlying autocall note by Monte Carlo."""
    if len(market.tickers) != 1:
        raise ValueError("Autocall is a single-name product (paper step 7)")

    obs_times = product.observation_times()
    # Simulate only at observation dates (plus 0 and maturity) — GBM with
    # piecewise-constant drift is exact on any grid, so the autocall logic is
    # unaffected and peak memory stays O(n_paths * n_obs).
    times, paths = simulate_paths(
        market=market,
        curve=curve,
        n_paths=n_paths,
        maturity_years=product.maturity_years,
        obs_times=obs_times,
        seed=seed,
    )
    S0 = market.spot[0]
    prices = paths[:, :, 0] / S0  # normalized to spot = 1
    obs_indices = np.searchsorted(times, obs_times)
    obs_indices = np.clip(obs_indices, 0, len(times) - 1)

    n_paths_real = prices.shape[0]
    active = np.ones(n_paths_real, dtype=bool)
    pv_total = np.zeros(n_paths_real, dtype=float)
    unpaid_coupons = np.zeros(n_paths_real, dtype=int)
    first_autocall_time = np.full(n_paths_real, product.maturity_years)

    for idx, t in zip(obs_indices, obs_times):
        s = prices[:, idx]
        df = float(curve.discount(t))
        pay_coupon = active & (s >= product.coupon_barrier)
        unpaid_coupons[active & ~pay_coupon] += 1
        if product.memory:
            n_pay = (unpaid_coupons + 1) * pay_coupon.astype(int)
            coupon_pv = product.par_value * product.coupon_rate * n_pay * df
            unpaid_coupons = np.where(pay_coupon, 0, unpaid_coupons)
        else:
            coupon_pv = product.par_value * product.coupon_rate * pay_coupon.astype(int) * df
        pv_total += coupon_pv

        trigger = active & (s >= product.autocall_barrier)
        first_autocall_time = np.where(trigger, t, first_autocall_time)
        principal_pv = product.par_value * trigger.astype(float) * df
        pv_total += principal_pv
        active = active & ~trigger

    # Maturity handling for paths that did not autocall
    df_T = float(curve.discount(product.maturity_years))
    s_T = prices[:, -1]
    survived = active
    above_protection = survived & (s_T >= product.protection_barrier)
    below_protection = survived & ~above_protection
    pv_total += product.par_value * above_protection.astype(float) * df_T
    pv_total += product.par_value * (s_T * below_protection.astype(float)) * df_T

    price = float(pv_total.mean())
    mc_stderr = float(pv_total.std(ddof=1) / np.sqrt(n_paths_real))
    autocall_prob = float(np.mean(first_autocall_time < product.maturity_years))
    expected_life = float(first_autocall_time.mean())
    loss_at_mat = float(np.mean(survived & (s_T < product.protection_barrier)))

    return {
        "product": product.name,
        "price": price,
        "mc_stderr": mc_stderr,
        "mc_ci95": (price - 1.96 * mc_stderr, price + 1.96 * mc_stderr),
        "expected_yield": float(np.log(price / product.par_value) / product.maturity_years)
            if price > 0 else float("nan"),
        "autocall_probability": autocall_prob,
        "expected_life_years": expected_life,
        "prob_capital_loss": loss_at_mat,
        "n_paths": n_paths_real,
        "n_observations": len(obs_times),
        "observation_times": obs_times.tolist(),
        "pv_sample": pv_total[:5000],
    }


def find_protection_for_target_yield(
    product_template: SyntheticProduct,
    market: MarketParams,
    curve: YieldCurve,
    target_yield: float,
    n_paths: int,
    seed: int = RANDOM_SEED,
    bracket: tuple[float, float] = (0.5, 1.0),
    tol: float = 1e-4,
    max_iter: int = 40,
) -> dict:
    """Solve for the capital-protection fraction giving a target expected yield.

    Implements the "Find barrier" helper (Section 1, step 9) by bisection on
    the ``protection`` field of the synthetic product. The search is monotonic
    because a higher protection increases the fair price and lowers the
    expected yield at a fixed participation.
    """
    lo, hi = bracket
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = SyntheticProduct(
            par_value=product_template.par_value,
            weights=product_template.weights,
            maturity_years=product_template.maturity_years,
            protection=mid,
            participation=product_template.participation,
            barriers=list(product_template.barriers),
            name=product_template.name,
        )
        r = price_synthetic(p, market, curve, n_paths=n_paths, seed=seed)
        y = r["expected_yield"]
        best = (mid, r)
        if abs(y - target_yield) < tol:
            break
        if y > target_yield:
            lo = mid
        else:
            hi = mid
    return {"protection": best[0], "result": best[1]}
