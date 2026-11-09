"""Experiment 2 — Historical backtest and stress testing of synthetic payoffs.

Walks forward through historical close prices to mark the structured note to
market, applies "Hit / Bump" stress scenarios, computes the specialized risk
metrics (VaR, gap risk, tracking error, whipsaw cost) and compares against the
Savings Benchmarking Ledger (Section 1, step 12).
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
from src.backtest import backtest_synthetic, whipsaw_cost

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true",
                   help="Use the paper configuration (identical to the default "
                        "for this experiment; included for symmetry).")
    return p.parse_args()


def main() -> dict:
    _ = _parse_args()

    bars, meta = load_ohlcv(
        config.UNIVERSE, config.SAMPLE_START, config.SAMPLE_END, timeframe="1d",
    )
    prices = wide_close(bars, config.UNIVERSE)
    weights = np.full(len(config.UNIVERSE), 1.0 / len(config.UNIVERSE))

    bt = backtest_synthetic(
        prices=prices,
        weights=weights,
        protection=config.SYNTHETIC_PROTECTION,
        participation=config.SYNTHETIC_PARTICIPATION,
        par_value=config.SYNTHETIC_PAR_VALUE,
        sales_charge=config.SALES_CHARGE,
        bank_rate=config.BANK_BENCHMARK_RATE,
        stress_bumps=(-0.20, -0.10, +0.10),
    )

    # Whipsaw cost from a simple drift-triggered rebalance log built from
    # the basket: the paper uses whipsaw in the context of hedge re-entry.
    basket = (prices.div(prices.iloc[0]) * weights).sum(axis=1)
    rets = basket.pct_change().fillna(0.0)
    trades = []
    pos = 0.0
    for date, r in rets.items():
        signal = np.sign(r)
        if signal == 0:
            continue
        qty = float(signal)
        if qty != pos:
            trades.append({"date": date, "asset": "basket", "quantity": qty - pos,
                           "price": float(basket.loc[date])})
            pos = qty
    ws = whipsaw_cost(pd.DataFrame(trades)) if trades else 0.0

    # Plots
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(bt.mtm.index, bt.mtm.values, label="Synthetic note MTM", lw=1.3)
    ax.plot(bt.bank.index, bt.bank.values, label="Bank note benchmark", lw=1.0,
            linestyle="--")
    ax.set_title("Historical mark-to-market vs. savings benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2_mtm_vs_bank.png", dpi=120)
    plt.close(fig)

    returns = bt.mtm.pct_change().dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(returns.values, bins=60, color="tab:orange", alpha=0.8)
    ax.set_title("Synthetic note daily returns distribution")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2_return_hist.png", dpi=120)
    plt.close(fig)

    summary = {
        "configuration": {
            "universe": config.UNIVERSE,
            "sample_start": config.SAMPLE_START,
            "sample_end": config.SAMPLE_END,
            "sales_charge": config.SALES_CHARGE,
            "bank_benchmark_rate": config.BANK_BENCHMARK_RATE,
            "stress_bumps": [-0.20, -0.10, +0.10],
            "ticker_details": meta.get("ticker_details"),
            "non_massive_source": meta.get("non_massive_source"),
        },
        "summary": bt.summary(),
        "stress_scenarios": bt.stress_scenarios,
        "whipsaw_cost": ws,
    }
    (RESULTS / "exp2_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Save MTM time series as CSV for traceability
    bt.mtm.rename("synthetic_mtm").to_frame().assign(
        bank=bt.bank.values,
    ).to_csv(RESULTS / "exp2_mtm_series.csv")

    lines = [
        "# Experiment 2 - Historical backtest",
        "",
        f"Sample: {config.SAMPLE_START} .. {config.SAMPLE_END}, "
        f"{len(prices)} trading days.",
        "",
        "## Risk and performance",
        f"- Initial value: {bt.summary()['initial_value']:.6f}",
        f"- Terminal value: {bt.summary()['terminal_value']:.6f}",
        f"- Cumulative return: {bt.summary()['cumulative_return']:.6f}",
        f"- Bank terminal value: {bt.summary()['bank_terminal_value']:.6f}",
        f"- Excess vs bank: {bt.summary()['excess_vs_bank']:.6f}",
        f"- VaR 95%: {bt.summary()['var_95']:.6f}",
        f"- Tracking error (annualized): {bt.summary()['tracking_error_ann']:.6f}",
        f"- Max daily drift: {bt.summary()['max_daily_drift']:.6f}",
        f"- Gap risk: {bt.summary()['gap_risk']:.6f}",
        f"- Whipsaw cost (basket reversal proxy): {ws:.6f}",
        "",
        "## Stress scenarios",
    ] + [f"- {k}: {v:.6f}" for k, v in bt.stress_scenarios.items()]
    (RESULTS / "exp2_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    main()
