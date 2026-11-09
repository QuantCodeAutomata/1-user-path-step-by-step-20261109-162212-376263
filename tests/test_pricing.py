"""Unit tests for the Monte Carlo pricing engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market import estimate_market_params
from src.pricing import (
    AutocallProduct,
    Barrier,
    MarketParams,
    SyntheticProduct,
    price_autocall,
    price_synthetic,
    simulate_paths,
)
from src.rates import flat_curve


@pytest.fixture
def curve():
    return flat_curve(0.05, as_of=pd.Timestamp("2024-01-01"))


@pytest.fixture
def market():
    return MarketParams(
        spot=np.array([100.0, 100.0]),
        vol=np.array([0.20, 0.25]),
        div_yield=np.array([0.0, 0.0]),
        correlation=np.array([[1.0, 0.3], [0.3, 1.0]]),
        tickers=["A", "B"],
    )


def test_simulate_paths_shapes_and_mean(market, curve):
    times, paths = simulate_paths(market, curve, n_paths=2000, maturity_years=1.0,
                                  steps_per_year=50, seed=1)
    assert paths.shape == (2000, 51, 2)
    assert times[0] == 0.0 and np.isclose(times[-1], 1.0)
    # Risk-neutral mean of terminal price is S0 * exp((r - q) * T)
    expected = market.spot * np.exp((0.05 - market.div_yield) * 1.0)
    assert np.allclose(paths[:, -1, :].mean(axis=0), expected, rtol=0.03)


def test_synthetic_price_bounds(market, curve):
    product = SyntheticProduct(
        par_value=100.0,
        weights=np.array([0.5, 0.5]),
        maturity_years=1.0,
        protection=1.0,
        participation=1.0,
        barriers=[],
    )
    res = price_synthetic(product, market, curve, n_paths=20_000, seed=7)
    # 100% capital protection => price must be >= 100 * exp(-r T)
    assert res["price"] >= 100.0 * np.exp(-0.05 * 1.0) - 1e-6
    # And bounded above by max basket value (no cap => bounded in practice only)
    assert res["mc_stderr"] > 0.0
    assert res["prob_loss"] == 0.0


def test_autocall_single_name_only(market, curve):
    product = AutocallProduct(
        par_value=100.0, maturity_years=1.0, obs_freq_months=3,
        autocall_barrier=1.0, coupon_barrier=0.7, protection_barrier=0.6,
        coupon_rate=0.02, memory=True,
    )
    with pytest.raises(ValueError):
        price_autocall(product, market, curve, n_paths=1000)


def test_autocall_price_positive(curve):
    single = MarketParams(
        spot=np.array([100.0]), vol=np.array([0.25]),
        div_yield=np.array([0.0]),
        correlation=np.array([[1.0]]), tickers=["A"],
    )
    product = AutocallProduct(
        par_value=100.0, maturity_years=2.0, obs_freq_months=6,
        autocall_barrier=1.0, coupon_barrier=0.7, protection_barrier=0.6,
        coupon_rate=0.03, memory=True,
    )
    res = price_autocall(product, single, curve, n_paths=10_000, seed=3)
    assert res["price"] > 0.0
    assert 0.0 <= res["autocall_probability"] <= 1.0
    assert 0.0 < res["expected_life_years"] <= 2.0


def test_barrier_payoff_direction():
    market = MarketParams(
        spot=np.array([100.0]), vol=np.array([0.20]),
        div_yield=np.array([0.0]),
        correlation=np.array([[1.0]]), tickers=["A"],
    )
    base = SyntheticProduct(par_value=100.0, weights=np.array([1.0]),
                            maturity_years=1.0, protection=1.0,
                            participation=1.0, barriers=[])
    bumped = SyntheticProduct(par_value=100.0, weights=np.array([1.0]),
                              maturity_years=1.0, protection=1.0,
                              participation=1.0,
                              barriers=[Barrier(level=1.1, direction=+1,
                                                participation=0.5)])
    curve = flat_curve(0.04)
    a = price_synthetic(base, market, curve, n_paths=20_000, seed=11)
    b = price_synthetic(bumped, market, curve, n_paths=20_000, seed=11)
    assert b["price"] > a["price"]


def test_estimate_market_params(synthetic_prices):
    mp = estimate_market_params(synthetic_prices, tickers=["A", "B", "C", "D", "E"])
    assert mp.vol.shape == (5,)
    assert np.all(mp.vol > 0)
    assert mp.correlation.shape == (5, 5)
    assert np.allclose(np.diag(mp.correlation), 1.0)
