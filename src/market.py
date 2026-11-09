"""Market parameter estimation from historical close prices.

Volatility, continuous dividend yield and correlation feed the Monte Carlo
pricing engine (Section 1, step 7 "Structure - basics" and step 11 "Price":
"Spot, Vol, Ctv, Correl").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pricing import MarketParams


def estimate_market_params(
    prices: pd.DataFrame,
    tickers: list[str],
    div_yield: np.ndarray | None = None,
    annualization: int = 252,
) -> MarketParams:
    """Estimate spot, volatility and correlation from daily close prices.

    Parameters
    ----------
    prices : wide DataFrame indexed by date, one column per ticker.
    tickers : explicit order of tickers (columns of the returned matrices).
    div_yield : optional per-asset continuous dividend yield (decimal). The
        paper labels this field "Ctv" (Section 1, step 7); without an input we
        default to zero.
    annualization : number of trading days in a year.
    """
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"missing tickers in prices: {missing}")
    px = prices[tickers].dropna()
    if len(px) < 20:
        raise ValueError("not enough observations to estimate market params")
    log_ret = np.log(px / px.shift(1)).dropna()
    vol = log_ret.std(ddof=1).to_numpy() * np.sqrt(annualization)
    correlation = log_ret.corr().to_numpy()
    spot = px.iloc[-1].to_numpy()
    if div_yield is None:
        div_yield = np.zeros(len(tickers))
    return MarketParams(
        spot=spot,
        vol=vol,
        div_yield=np.asarray(div_yield, dtype=float),
        correlation=correlation,
        tickers=tickers,
    )
