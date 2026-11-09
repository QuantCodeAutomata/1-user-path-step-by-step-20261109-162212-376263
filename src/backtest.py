"""Historical backtest and stress testing of structured payoffs.

Reproduces the Backtest screen described in Section 1, step 12 "Backtest" of
the SmartDelta user path. The screen specifies a mark-to-market walk over
daily close prices, deduction of a sales charge, "No / Coupon Bumps" stress
scenarios, and the risk cost metrics (tracking error, whipsaw cost, gap risk,
max daily drift, VaR), together with the "Savings Benchmarking Ledger" against
a bank note.

The precise definitions of "whipsaw" and "gap risk" are not standardized in
the paper (Section 4 marks several of these as "to clarify"). This module
implements each one explicitly, with the interpretation documented inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    BANK_BENCHMARK_RATE,
    GAP_RISK_WINDOW,
    SALES_CHARGE,
    STRESS_BUMP_DAY_FRACTION,
    STRESS_BUMP_PCT,
    VAR_CONFIDENCE,
)


def apply_bump(prices: pd.DataFrame, bump_pct: float,
               day_fraction: float = STRESS_BUMP_DAY_FRACTION) -> pd.DataFrame:
    """Return a copy of ``prices`` with a one-off ``bump_pct`` price shock.

    The shock is applied multiplicatively from the stress date onward; this
    implements the "Hit / Bump" stress scenario of the Backtest screen; the
    paper does not define the shock mathematically so an instant multiplicative
    reset is used (Section 4 "to clarify").
    """
    if not 0.0 <= day_fraction <= 1.0:
        raise ValueError("day_fraction must be in [0, 1]")
    shocked = prices.copy()
    idx = int(len(prices) * day_fraction)
    shocked.iloc[idx:] = shocked.iloc[idx:] * (1.0 + bump_pct)
    return shocked


def basket_prices(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted basket value normalized to 1 at t=0 (same weights as pricing)."""
    if prices.shape[1] != len(weights):
        raise ValueError("prices columns and weights length mismatch")
    normalized = prices.divide(prices.iloc[0])
    return (normalized * weights).sum(axis=1)


def synthetic_mtm(
    basket: pd.Series,
    protection: float,
    participation: float,
    par_value: float,
    sales_charge: float = SALES_CHARGE,
) -> pd.Series:
    """Daily mark-to-market value of a synthetic capital-protected note.

    At each date, MTM = par * (protection + participation * max(basket - 1, 0))
    with the sales charge deducted from the initial principal.
    """
    initial_cash = par_value * (1.0 - sales_charge)
    base = protection + participation * np.maximum(basket - 1.0, 0.0)
    mtm = initial_cash * base / base.iloc[0]
    return mtm


def autocall_mtm(
    prices: pd.Series,
    obs_dates: pd.DatetimeIndex,
    autocall_barrier: float,
    coupon_barrier: float,
    protection_barrier: float,
    coupon_rate: float,
    memory: bool,
    par_value: float,
    sales_charge: float = SALES_CHARGE,
) -> tuple[pd.Series, dict]:
    """Walk-forward MTM for an autocall note on a single underlying."""
    s0 = prices.iloc[0]
    cash_init = par_value * (1.0 - sales_charge)
    mtm = pd.Series(cash_init, index=prices.index, dtype=float)
    unpaid = 0
    autocalled_on: pd.Timestamp | None = None
    cumulative_coupon = 0.0

    def live_value(date: pd.Timestamp) -> float:
        s = prices.loc[date] / s0
        remaining = max(1e-6, (prices.index[-1] - date).days / 365.25)
        # Approximation: live value = par discounted at coupon rate adjusted by
        # observed cushion to the protection barrier. The paper does not pin a
        # mark-to-market formula so we use a cushion-weighted ``par`` value.
        cushion = max(0.0, (s - protection_barrier) / (1.0 - protection_barrier))
        return par_value * (0.5 + 0.5 * cushion) + cumulative_coupon

    for date in prices.index:
        if autocalled_on is not None:
            mtm.loc[date] = par_value + cumulative_coupon
            continue
        s = prices.loc[date] / s0
        if date in obs_dates:
            if s >= autocall_barrier:
                autocalled_on = date
                coupon_now = coupon_rate * par_value * (1 + (unpaid if memory else 0))
                cumulative_coupon += coupon_now
                unpaid = 0
                mtm.loc[date] = par_value + cumulative_coupon
                continue
            if s >= coupon_barrier:
                coupon_now = coupon_rate * par_value * (1 + (unpaid if memory else 0))
                cumulative_coupon += coupon_now
                unpaid = 0
            else:
                unpaid += 1
        mtm.loc[date] = live_value(date)

    if autocalled_on is None:
        # Terminal payoff at maturity
        final_s = prices.iloc[-1] / s0
        if final_s >= protection_barrier:
            terminal = par_value + cumulative_coupon
        else:
            terminal = par_value * final_s
        mtm.iloc[-1] = terminal

    return mtm, {
        "autocalled_on": autocalled_on,
        "cumulative_coupon": cumulative_coupon,
        "terminal_value": float(mtm.iloc[-1]),
    }


# -----------------------------------------------------------------------------
# Risk metrics
# -----------------------------------------------------------------------------


def value_at_risk(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """Historical Value-at-Risk at ``confidence``. Positive number is a loss."""
    q = np.quantile(returns.dropna(), 1.0 - confidence)
    return float(-q)


def tracking_error(strategy: pd.Series, target: pd.Series) -> float:
    """Annualized tracking error between strategy and target daily returns."""
    r_s = strategy.pct_change().dropna()
    r_t = target.pct_change().dropna()
    aligned = pd.concat([r_s, r_t], axis=1, join="inner").dropna()
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(diff.std(ddof=1) * np.sqrt(252))


def max_daily_drift(strategy: pd.Series, target: pd.Series) -> float:
    """Largest absolute single-day relative drift between strategy and target."""
    aligned = pd.concat([strategy, target], axis=1, join="inner").dropna()
    return float(((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs() /
                  aligned.iloc[:, 1].abs()).max())


def gap_risk(returns: pd.Series, window: int = GAP_RISK_WINDOW) -> float:
    """Worst single-day loss (positive number) seen over the trailing window.

    The paper lists "gap risk" without a formula (Section 4 "to clarify").
    Interpreted here as the worst overnight drawdown over the backtest window,
    which is the risk metric referenced by the Dashboard drift alerts.
    """
    r = returns.dropna()
    if window > 0 and len(r) > window:
        r = r.iloc[-window:]
    return float(-r.min())


def whipsaw_cost(trades: pd.DataFrame, lookback_days: int = 5) -> float:
    """Sum of trading-loss "whipsaw" costs from quick reversal trades.

    The paper lists "whipsaw cost" without a formula; this interpretation
    matches the common industry definition: for every pair of trades on the
    same asset where the direction reverses within ``lookback_days``, the
    adverse price movement between them is treated as a whipsaw cost.
    """
    if trades.empty:
        return 0.0
    required = {"date", "asset", "quantity", "price"}
    if not required.issubset(trades.columns):
        raise ValueError(f"trades must contain columns {required}")
    cost = 0.0
    for asset, group in trades.sort_values("date").groupby("asset"):
        prev_date = None
        prev_sign = 0
        prev_price = None
        prev_qty = 0.0
        for _, row in group.iterrows():
            sign = int(np.sign(row["quantity"]))
            if (prev_sign != 0 and sign != 0 and sign != prev_sign and prev_date is not None
                    and (row["date"] - prev_date).days <= lookback_days):
                qty_reversed = min(abs(prev_qty), abs(row["quantity"]))
                cost += qty_reversed * abs(row["price"] - prev_price)
            prev_date = row["date"]
            prev_sign = sign
            prev_price = row["price"]
            prev_qty = row["quantity"]
    return float(cost)


def bank_benchmark(initial: float, dates: pd.DatetimeIndex,
                   annual_rate: float = BANK_BENCHMARK_RATE) -> pd.Series:
    """Synthetic bank-note value compounding at ``annual_rate`` daily.

    Section 1, step 12 calls this the "Savings Benchmarking Ledger"; the
    bank note is the comparison baseline.
    """
    days = (dates - dates[0]).days.to_numpy()
    return pd.Series(initial * np.exp(annual_rate * days / 365.25), index=dates)


# -----------------------------------------------------------------------------
# Full backtest orchestrator
# -----------------------------------------------------------------------------


@dataclass
class BacktestResult:
    mtm: pd.Series
    metrics: dict
    bank: pd.Series
    stress_scenarios: dict

    def summary(self) -> dict:
        return {
            "initial_value": float(self.mtm.iloc[0]),
            "terminal_value": float(self.mtm.iloc[-1]),
            "cumulative_return": float(self.mtm.iloc[-1] / self.mtm.iloc[0] - 1.0),
            "bank_terminal_value": float(self.bank.iloc[-1]),
            "excess_vs_bank": float(self.mtm.iloc[-1] - self.bank.iloc[-1]),
            **self.metrics,
        }


def backtest_synthetic(
    prices: pd.DataFrame,
    weights: np.ndarray,
    protection: float,
    participation: float,
    par_value: float,
    sales_charge: float = SALES_CHARGE,
    bank_rate: float = BANK_BENCHMARK_RATE,
    stress_bumps: Iterable[float] = (STRESS_BUMP_PCT,),
) -> BacktestResult:
    """Full historical backtest of a synthetic payoff on a basket."""
    if prices.empty:
        raise ValueError("prices is empty")

    basket = basket_prices(prices, weights)
    mtm = synthetic_mtm(basket, protection, participation, par_value, sales_charge)
    returns = mtm.pct_change()
    target = basket * mtm.iloc[0]  # the target exposure that the hedge tracks

    scenarios: dict[str, float] = {}
    for b in stress_bumps:
        shocked_basket = basket_prices(apply_bump(prices, b), weights)
        shocked_mtm = synthetic_mtm(shocked_basket, protection, participation,
                                    par_value, sales_charge)
        scenarios[f"bump_{int(b*100):+d}pct_terminal"] = float(shocked_mtm.iloc[-1])

    metrics = {
        "var_95": value_at_risk(returns, VAR_CONFIDENCE),
        "tracking_error_ann": tracking_error(mtm, target),
        "max_daily_drift": max_daily_drift(mtm, target),
        "gap_risk": gap_risk(returns, GAP_RISK_WINDOW),
        "sales_charge": sales_charge,
    }

    bank = bank_benchmark(float(mtm.iloc[0]), prices.index, bank_rate)
    return BacktestResult(mtm=mtm, metrics=metrics, bank=bank, stress_scenarios=scenarios)
