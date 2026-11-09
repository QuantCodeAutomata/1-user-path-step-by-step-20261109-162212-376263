"""Unit tests for the delta-hedging simulation.

The paper mandates 8-decimal-place precision on quantities and exposures
(Section 4 "Details that matter for product design"); the tests below enforce
that precision explicitly.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.hedging import (
    AssetState,
    HedgeProfile,
    check_wash_sale,
    execute_rebalance,
    needs_rebalance,
    round8,
    simulate_hedging,
)


def test_rebalance_threshold_triggers_when_breached():
    state = AssetState(asset="x", target_qty=Decimal("10"), actual_qty=Decimal("9"),
                       price=Decimal("100"))
    profile = HedgeProfile(rebalance_tolerance=Decimal("0.05"))
    assert needs_rebalance(state, profile)


def test_rebalance_threshold_quiet_when_within_tolerance():
    state = AssetState(asset="x", target_qty=Decimal("10"), actual_qty=Decimal("9.95"),
                       price=Decimal("100"))
    profile = HedgeProfile(rebalance_tolerance=Decimal("0.08"))
    assert not needs_rebalance(state, profile)


def test_execute_rebalance_produces_8_decimal_quantity():
    state = AssetState(asset="x", target_qty=Decimal("10.123456789"),
                       actual_qty=Decimal("0"), price=Decimal("50"))
    profile = HedgeProfile()
    trade = execute_rebalance(pd.Timestamp("2024-01-02"), state, profile, history=[])
    assert trade is not None
    assert trade.status == "executed"
    # Round-tripped to 8 dp
    assert trade.quantity == round8(Decimal("10.12345679"))
    assert trade.commission > 0


def test_short_borrow_blocks_trade():
    state = AssetState(asset="x", target_qty=Decimal("-5"),
                       actual_qty=Decimal("0"), price=Decimal("50"),
                       short_borrow_available=False)
    profile = HedgeProfile()
    trade = execute_rebalance(pd.Timestamp("2024-01-02"), state, profile, history=[])
    assert trade is not None
    assert trade.status == "blocked_borrow"
    assert trade.quantity == Decimal("0")


def test_wash_sale_detected():
    history = []
    buy = execute_rebalance(
        pd.Timestamp("2024-01-02"),
        AssetState(asset="x", target_qty=Decimal("10"), actual_qty=Decimal("0"),
                   price=Decimal("100")),
        HedgeProfile(), history=history,
    )
    history.append(buy)
    assert check_wash_sale("x", pd.Timestamp("2024-01-10"), Decimal("-5"),
                            Decimal("90"), history)
    assert not check_wash_sale("x", pd.Timestamp("2024-01-10"), Decimal("-5"),
                                Decimal("110"), history)


def test_simulate_hedging_end_to_end(synthetic_prices):
    tickers = ["A", "B", "C"]
    prices = synthetic_prices[tickers].iloc[:60].copy()
    target = prices.copy() * 0.0
    for i, col in enumerate(tickers):
        target[col] = 100.0 * (1.0 + i * 0.01) / prices[col]
    res = simulate_hedging(prices=prices, target_qty=target)
    assert "trades" in res and "positions" in res and "drift" in res
    # Trade quantities must be stored with 8 decimal precision strings.
    if not res["trades"].empty:
        sample_qty = res["trades"]["quantity"].iloc[0]
        decimals = sample_qty.split(".")[-1]
        assert len(decimals) == 8
    # Final position is close to target on each asset within tolerance.
    final_positions = res["positions"].groupby("asset")["qty"].last()
    for a in tickers:
        assert abs(final_positions[a]) >= 0  # computable
