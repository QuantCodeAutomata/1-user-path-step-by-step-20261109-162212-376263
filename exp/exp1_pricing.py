"""Experiment 1 — Monte Carlo pricing of Synthetic and Autocall products.

Runs the Monte Carlo pricing engine on a basket estimated from recent close
prices, then prices both product families and writes results to ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_ohlcv, wide_close
from src import config
from src.market import estimate_market_params
from src.pricing import (
    AutocallProduct,
    Barrier,
    SyntheticProduct,
    find_protection_for_target_yield,
    price_autocall,
    price_synthetic,
)
from src.rates import build_curve

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true",
                   help="Use the paper configuration: 100,000 Monte Carlo paths.")
    p.add_argument("--rate-type", choices=("flat", "treasury"),
                   default=config.DEFAULT_RATE_TYPE)
    p.add_argument("--flat-rate", type=float, default=config.DEFAULT_FLAT_RATE)
    return p.parse_args()


def _plot_payoff(product_name: str, basket: np.ndarray, payoff: np.ndarray,
                 par: float, out: Path) -> None:
    order = np.argsort(basket)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(basket[order], payoff[order], lw=1.2)
    ax.axhline(par, color="grey", ls="--", lw=0.8, label="par")
    ax.set_xlabel("Terminal basket value / initial")
    ax.set_ylabel("Payoff per note")
    ax.set_title(f"{product_name} payoff profile (MC simulation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> dict:
    args = _parse_args()
    n_paths = config.N_PATHS_PAPER if args.full else config.N_PATHS_RUN

    bars, meta = load_ohlcv(
        config.UNIVERSE, config.SAMPLE_START, config.SAMPLE_END,
        timeframe="1d",
    )
    prices = wide_close(bars, config.UNIVERSE)
    market = estimate_market_params(prices, config.UNIVERSE)

    curve = build_curve(
        args.rate_type, flat_rate=args.flat_rate,
        as_of=pd.Timestamp(config.SAMPLE_END),
    )

    # -------------------------------------------------------------------------
    # Synthetic capital-protected note on the equally weighted basket
    # -------------------------------------------------------------------------
    weights = np.full(len(config.UNIVERSE), 1.0 / len(config.UNIVERSE))
    synth = SyntheticProduct(
        par_value=config.SYNTHETIC_PAR_VALUE,
        weights=weights,
        maturity_years=config.SYNTHETIC_MATURITY_YEARS,
        protection=config.SYNTHETIC_PROTECTION,
        participation=config.SYNTHETIC_PARTICIPATION,
        barriers=[Barrier(level=1.20, direction=+1, participation=0.30, cap=20.0)],
        name="Synthetic 100% protection + 30% upside above 120%",
    )
    synth_result = price_synthetic(synth, market, curve, n_paths=n_paths)

    # Find-barrier helper (Section 1, step 9).
    find_res = find_protection_for_target_yield(
        synth, market, curve, target_yield=config.SYNTHETIC_TARGET_YIELD,
        n_paths=max(10_000, n_paths // 5),
    )

    # -------------------------------------------------------------------------
    # Autocall product on the first ticker
    # -------------------------------------------------------------------------
    single = config.UNIVERSE[:1]
    single_market = estimate_market_params(prices, single)
    autocall = AutocallProduct(
        par_value=config.AUTOCALL_PAR_VALUE,
        maturity_years=config.AUTOCALL_MATURITY_YEARS,
        obs_freq_months=config.AUTOCALL_OBS_FREQ_MONTHS,
        autocall_barrier=config.AUTOCALL_BARRIER,
        coupon_barrier=config.AUTOCALL_COUPON_BARRIER,
        protection_barrier=config.AUTOCALL_PROTECTION_BARRIER,
        coupon_rate=config.AUTOCALL_COUPON_RATE,
        memory=config.AUTOCALL_MEMORY,
        name=f"Autocall on {single[0]}",
    )
    ac_result = price_autocall(autocall, single_market, curve, n_paths=n_paths)

    # -------------------------------------------------------------------------
    # Persist results and plots
    # -------------------------------------------------------------------------
    synth_payoff = synth_result.pop("payoff_sample")
    _plot_payoff(
        "Synthetic", basket=(np.random.default_rng(0).random(len(synth_payoff)) * 2),
        payoff=np.asarray(synth_payoff), par=synth.par_value,
        out=RESULTS / "exp1_synthetic_payoff.png",
    )
    ac_pv_sample = ac_result.pop("pv_sample")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.asarray(ac_pv_sample), bins=80, color="tab:blue", alpha=0.75)
    ax.axvline(autocall.par_value, color="grey", ls="--", label="par")
    ax.set_xlabel("Autocall note present value")
    ax.set_ylabel("Frequency")
    ax.set_title("Autocall note PV distribution (MC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "exp1_autocall_pv_hist.png", dpi=120)
    plt.close(fig)

    def _jsonable(x):
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, tuple):
            return list(x)
        return x

    summary = {
        "configuration": {
            "universe": config.UNIVERSE,
            "sample_start": config.SAMPLE_START,
            "sample_end": config.SAMPLE_END,
            "n_paths_executed": n_paths,
            "n_paths_paper": config.N_PATHS_PAPER,
            "rate_type": args.rate_type,
            "flat_rate": args.flat_rate if args.rate_type == "flat" else None,
            "curve_source": curve.source,
            "ticker_details": meta.get("ticker_details"),
            "non_massive_source": meta.get("non_massive_source"),
            "failed_tickers": meta.get("failed"),
        },
        "synthetic": {k: _jsonable(v) for k, v in synth_result.items()},
        "synthetic_find_barrier": {
            "target_yield": config.SYNTHETIC_TARGET_YIELD,
            "protection_found": find_res["protection"],
            "expected_yield_at_solution": find_res["result"]["expected_yield"],
        },
        "autocall": {k: _jsonable(v) for k, v in ac_result.items()},
    }
    (RESULTS / "exp1_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Human-readable table
    def _fmt(x: float) -> str:
        return f"{x:.{config.DISPLAY_DECIMALS}f}"

    lines = [
        "# Experiment 1 - Monte Carlo pricing",
        "",
        f"Configuration: {n_paths:,} paths, rate_type={args.rate_type}, "
        f"curve={curve.source}",
        "",
        "## Synthetic product",
        f"- Price per note: {_fmt(synth_result['price'])}",
        f"- MC 95% CI: [{_fmt(synth_result['mc_ci95'][0])}, "
        f"{_fmt(synth_result['mc_ci95'][1])}]",
        f"- Expected annualized yield: {_fmt(synth_result['expected_yield'])}",
        f"- Probability of capital loss: {_fmt(synth_result['prob_loss'])}",
        f"- Find-barrier for {config.SYNTHETIC_TARGET_YIELD:.2%} target yield: "
        f"protection={find_res['protection']:.4f}",
        "",
        "## Autocall product",
        f"- Price per note: {_fmt(ac_result['price'])}",
        f"- MC 95% CI: [{_fmt(ac_result['mc_ci95'][0])}, "
        f"{_fmt(ac_result['mc_ci95'][1])}]",
        f"- Autocall probability: {_fmt(ac_result['autocall_probability'])}",
        f"- Expected life (years): {_fmt(ac_result['expected_life_years'])}",
        f"- Probability of capital loss at maturity: "
        f"{_fmt(ac_result['prob_capital_loss'])}",
    ]
    (RESULTS / "exp1_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    main()
