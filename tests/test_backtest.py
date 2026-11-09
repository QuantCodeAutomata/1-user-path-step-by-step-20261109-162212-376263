"""Unit tests for the backtest and risk-metric module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    apply_bump,
    backtest_synthetic,
    gap_risk,
    tracking_error,
    value_at_risk,
    whipsaw_cost,
)


def test_apply_bump_shifts_prices(synthetic_prices):
    bumped = apply_bump(synthetic_prices, -0.1, day_fraction=0.5)
    idx = int(len(synthetic_prices) * 0.5)
    assert np.allclose(bumped.iloc[idx:] / synthetic_prices.iloc[idx:], 0.9)
    assert np.allclose(bumped.iloc[:idx], synthetic_prices.iloc[:idx])


def test_var_direction(synthetic_prices):
    returns = synthetic_prices.iloc[:, 0].pct_change().dropna()
    v = value_at_risk(returns, 0.95)
    assert v > 0
    assert v < 0.2  # sanity bound for ~25% vol daily data


def test_tracking_error_zero_for_identical():
    s = pd.Series(np.linspace(100, 110, 60),
                  index=pd.bdate_range("2023-01-02", periods=60))
    assert tracking_error(s, s) == pytest.approx(0.0, abs=1e-12)


def test_gap_risk_positive():
    r = pd.Series([0.01, -0.03, 0.02, -0.07, 0.01])
    assert gap_risk(r, window=10) == pytest.approx(0.07)


def test_whipsaw_cost_detects_reversal():
    trades = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-02"), "asset": "x",
         "quantity": 10.0, "price": 100.0},
        {"date": pd.Timestamp("2024-01-03"), "asset": "x",
         "quantity": -10.0, "price": 97.0},
    ])
    cost = whipsaw_cost(trades)
    assert cost == pytest.approx(30.0)


def test_backtest_synthetic_returns_consistent_summary(synthetic_prices):
    weights = np.full(5, 0.2)
    result = backtest_synthetic(
        prices=synthetic_prices,
        weights=weights,
        protection=1.0,
        participation=1.0,
        par_value=100.0,
        sales_charge=0.005,
    )
    s = result.summary()
    assert s["initial_value"] < 100.0  # sales charge deducted
    assert s["terminal_value"] > 0
    assert "var_95" in s and s["var_95"] >= 0
    assert "tracking_error_ann" in s and s["tracking_error_ann"] >= 0
    assert result.bank.iloc[0] == pytest.approx(s["initial_value"])
