"""Experiment 3 — Dynamic delta hedging and fractional rebalancing.

Simulates the Delta Hub rebalancing engine with 8-decimal-place precision and
full execution frictions (Section 1 steps 23-28 and Section 4 "Hedge profile").
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
from src.hedging import HedgeProfile, simulate_hedging

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true",
                   help="Use the paper configuration (full universe, default here).")
    return p.parse_args()


def _rolling_target(prices: pd.DataFrame, par_value: float,
                    protection: float, participation: float) -> pd.DataFrame:
    """Build a rolling target delta per asset derived from a synthetic note.

    Target quantity per asset ``i`` at time ``t``:
        qty_i(t) = participation * par * weight_i * basket_sensitivity(t) / price_i(t)
    where ``basket_sensitivity`` is 1 when the basket is above its initial value
    (participation in upside) and zero otherwise. This matches the delta of the
    synthetic payoff implemented in :mod:`src.pricing`.
    """
    n = prices.shape[1]
    weights = np.full(n, 1.0 / n)
    basket = (prices.div(prices.iloc[0]) * weights).sum(axis=1)
    sens = np.where(basket > 1.0, 1.0, 0.0)
    # Target notional per asset
    target_notional_per_asset = participation * par_value / n
    target_qty = prices.copy() * 0.0
    for col in prices.columns:
        target_qty[col] = target_notional_per_asset * sens / prices[col]
    return target_qty


def main() -> dict:
    _ = _parse_args()

    bars, meta = load_ohlcv(
        config.UNIVERSE, config.SAMPLE_START, config.SAMPLE_END, timeframe="1d",
    )
    prices = wide_close(bars, config.UNIVERSE)
    target_qty = _rolling_target(
        prices, par_value=config.SYNTHETIC_PAR_VALUE,
        protection=config.SYNTHETIC_PROTECTION,
        participation=config.SYNTHETIC_PARTICIPATION,
    )

    profile = HedgeProfile()
    result = simulate_hedging(prices=prices, target_qty=target_qty, profile=profile)

    # Plots
    pv = result["portfolio_value"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pv.index, pv.values, lw=1.2, label="Portfolio value (hedged)")
    ax.set_title("Delta-hedged portfolio value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "exp3_portfolio_value.png", dpi=120)
    plt.close(fig)

    drift_df = result["drift"]
    worst = drift_df.groupby("date")["drift_pct"].max()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(worst.index, worst.values, lw=0.9, color="tab:red")
    ax.axhline(float(profile.rebalance_tolerance), color="grey", ls="--",
               label="Tolerance")
    ax.set_title("Worst drift across assets vs. rebalance tolerance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drift (fraction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "exp3_drift.png", dpi=120)
    plt.close(fig)

    result["trades"].to_csv(RESULTS / "exp3_trades.csv", index=False)
    result["drift"].to_csv(RESULTS / "exp3_drift.csv", index=False)
    result["positions"].to_csv(RESULTS / "exp3_positions.csv", index=False)

    summary = {
        "configuration": {
            "universe": config.UNIVERSE,
            "sample_start": config.SAMPLE_START,
            "sample_end": config.SAMPLE_END,
            "rebalance_tolerance_pct": config.REBALANCE_TOLERANCE_PCT,
            "commission_rate": config.COMMISSION_RATE,
            "slippage_rate": config.SLIPPAGE_RATE,
            "short_borrow_rate": config.SHORT_BORROW_RATE,
            "wash_sale_window_days": config.WASH_SALE_WINDOW_DAYS,
            "ticker_details": meta.get("ticker_details"),
            "non_massive_source": meta.get("non_massive_source"),
        },
        "summary": result["summary"],
    }
    (RESULTS / "exp3_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Experiment 3 - Delta hedging",
        "",
        f"Sample: {config.SAMPLE_START} .. {config.SAMPLE_END}",
        "",
        "## Execution summary",
        f"- Trades executed: {result['summary']['n_trades']}",
        f"- Trades blocked (short borrow): {result['summary']['n_blocked_borrow']}",
        f"- Wash sales flagged: {result['summary']['n_wash_sales']}",
        f"- Total commission: {result['summary']['total_commission']:.8f}",
        f"- Total slippage: {result['summary']['total_slippage']:.8f}",
        f"- Total short borrow fees: {result['summary']['total_borrow_fee']:.8f}",
        f"- Total tax: {result['summary']['total_tax']:.8f}",
        f"- Final cash: {result['summary']['final_cash']:.8f}",
        f"- Final portfolio value: {result['summary']['final_portfolio_value']:.8f}",
    ]
    (RESULTS / "exp3_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    main()
