"""Synthetic market data used by tests and by experiment 3's demo run.

Creates a correlated geometric Brownian motion basket in the same tidy format
returned by :mod:`data.data_loader`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    spot: np.ndarray | None = None,
    vol: np.ndarray | None = None,
    drift: np.ndarray | None = None,
    correlation: np.ndarray | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate correlated GBM close prices on the business calendar.

    Returns a DataFrame indexed by trading date, one column per ticker.
    """
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    k = len(tickers)
    if spot is None:
        spot = np.full(k, 100.0)
    if vol is None:
        vol = np.full(k, 0.20)
    if drift is None:
        drift = np.full(k, 0.05)
    if correlation is None:
        correlation = 0.3 * np.ones((k, k)) + 0.7 * np.eye(k)

    L = np.linalg.cholesky(correlation)
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    z = rng.standard_normal(size=(n - 1, k)) @ L.T
    log_ret = (drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
    log_price = np.vstack([np.log(spot), np.log(spot) + np.cumsum(log_ret, axis=0)])
    prices = np.exp(log_price)
    return pd.DataFrame(prices, index=dates, columns=tickers)
