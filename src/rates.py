"""Interest-rate term structure for the pricing engine.

The SmartDelta pricing screen (Section 1, step 11 "Price") lets the user choose
either a flat rate or a curve ("Index Type -> Fetch Curve"). Section 4
"Details that matter for product design" names US Treasury as the curve source.

This module exposes two constructors returning a callable
``discount_factor(t)`` plus the zero-rate curve used:

* ``flat_curve(rate)``        — constant continuously compounded zero rate
* ``treasury_curve(as_of)``   — zero rates bootstrapped from FRED's constant
                                  maturity series (DGS1MO..DGS30)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .config import TREASURY_FRED_SERIES, TREASURY_TENORS_YEARS


@dataclass
class YieldCurve:
    """Continuous zero-rate term structure.

    Attributes
    ----------
    tenors : array of years to maturity (ascending)
    zero_rates : array of continuously compounded zero rates (decimals)
    source : descriptive source string
    as_of : curve observation date
    """

    tenors: np.ndarray
    zero_rates: np.ndarray
    source: str
    as_of: pd.Timestamp

    def zero(self, t: float | np.ndarray) -> np.ndarray:
        """Linear interpolation of the zero rate at time ``t`` (years)."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        if t_arr.min() < 0.0:
            raise ValueError("Time to maturity must be non-negative")
        return np.interp(t_arr, self.tenors, self.zero_rates)

    def discount(self, t: float | np.ndarray) -> np.ndarray:
        """Discount factor P(0, t) = exp(-r(t) * t) under continuous compounding."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.exp(-self.zero(t_arr) * t_arr)

    def forward(self, t1: float, t2: float) -> float:
        """Simple forward rate from t1 to t2 (continuously compounded)."""
        if t2 <= t1:
            raise ValueError("t2 must be strictly greater than t1")
        r1, r2 = float(self.zero(t1)), float(self.zero(t2))
        return (r2 * t2 - r1 * t1) / (t2 - t1)


def flat_curve(rate: float, as_of: pd.Timestamp | None = None) -> YieldCurve:
    """Flat continuously compounded zero curve at ``rate``."""
    if as_of is None:
        as_of = pd.Timestamp.today().normalize()
    tenors = np.array([1.0 / 365.0, 30.0], dtype=float)
    rates = np.array([rate, rate], dtype=float)
    return YieldCurve(tenors=tenors, zero_rates=rates, source="flat", as_of=as_of)


def _fetch_fred_series(series: Sequence[str], as_of: pd.Timestamp) -> pd.Series:
    """Fetch the most recent available value for each FRED series on or before
    ``as_of``. Uses pandas_datareader's FRED reader; the optional FRED_API_KEY
    environment variable is honored when present.
    """
    from pandas_datareader import data as pdr  # local import keeps module light

    start = (as_of - timedelta(days=30)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")
    api_key = os.environ.get("FRED_API_KEY")
    df = pdr.DataReader(list(series), "fred", start=start, end=end, api_key=api_key)
    df = df.ffill().dropna(how="all")
    if df.empty:
        raise RuntimeError(
            f"FRED returned no data for series={list(series)} between {start} and {end}"
        )
    latest = df.iloc[-1]
    return latest


def treasury_curve(as_of: pd.Timestamp | None = None) -> YieldCurve:
    """US Treasury zero curve bootstrapped from FRED constant-maturity series.

    FRED publishes par (coupon-equivalent) yields. For tenors <= 1Y they are
    treated as zero rates on a continuously compounded basis after converting
    the quoted APR to continuous compounding. For longer tenors, a simple
    par-to-zero bootstrap is applied assuming annual coupons and the
    continuous-compounding convention used by the pricing engine (Section 1,
    step 11 of the paper is silent on the exact bootstrap).
    """
    if as_of is None:
        as_of = pd.Timestamp.today().normalize()

    latest = _fetch_fred_series(TREASURY_FRED_SERIES, as_of)

    tenors = np.asarray(TREASURY_TENORS_YEARS, dtype=float)
    # FRED yields are in percent; convert to decimal.
    par_yields = latest.to_numpy(dtype=float) / 100.0

    if np.isnan(par_yields).any():
        # Interpolate any missing tenors against the available ones.
        valid = ~np.isnan(par_yields)
        par_yields = np.interp(tenors, tenors[valid], par_yields[valid])

    # Convert quoted APR (semi-annual compounding for >=1Y treasuries) to
    # continuously compounded equivalent, then bootstrap.
    cc_par = 2.0 * np.log1p(par_yields / 2.0)
    zero_rates = np.empty_like(cc_par)

    short = tenors <= 1.0
    zero_rates[short] = cc_par[short]

    for i, t in enumerate(tenors):
        if t <= 1.0:
            continue
        c = par_yields[i]
        # Coupon dates at integer years up to t; irregular front stub assumed
        # to be 1 year back from maturity when t is non-integer.
        dates = np.arange(int(np.ceil(t)) - int(np.floor(t)) + 1, int(np.floor(t)) + 1)
        coupon_times = np.arange(1.0, np.floor(t) + 1.0)
        if coupon_times.size == 0 or coupon_times[-1] < t - 1e-9:
            coupon_times = np.append(coupon_times, t)

        # Discount factors at known tenors by interpolating ``zero_rates`` filled
        # so far.
        known_t = tenors[:i + 1]
        known_z = np.where(np.arange(i + 1) == i, cc_par[i], zero_rates[:i + 1])
        # Solve for z_i such that sum_coupons + principal == par (price=1).
        def price(z_i: float) -> float:
            z_eff = known_z.copy()
            z_eff[i] = z_i
            pv = 0.0
            for ct in coupon_times:
                z_ct = float(np.interp(ct, known_t, z_eff))
                pv += c * np.exp(-z_ct * ct)
            z_T = float(np.interp(t, known_t, z_eff))
            pv += np.exp(-z_T * t)
            return pv - 1.0

        lo, hi = -0.10, 0.50
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if price(mid) > 0:
                lo = mid
            else:
                hi = mid
        zero_rates[i] = 0.5 * (lo + hi)

    return YieldCurve(
        tenors=tenors,
        zero_rates=zero_rates,
        source="FRED US Treasury",
        as_of=pd.Timestamp(as_of),
    )


def build_curve(rate_type: str, flat_rate: float | None = None,
                as_of: pd.Timestamp | None = None) -> YieldCurve:
    """Factory used by the experiments.

    Parameters
    ----------
    rate_type : "flat" or "treasury"
    flat_rate : used when ``rate_type`` is "flat"
    as_of : observation date for the treasury curve
    """
    if rate_type == "flat":
        if flat_rate is None:
            raise ValueError("flat_rate is required for a flat curve")
        return flat_curve(flat_rate, as_of=as_of)
    if rate_type == "treasury":
        return treasury_curve(as_of=as_of)
    raise ValueError(f"Unknown rate_type: {rate_type!r}")


def discount_callable(curve: YieldCurve) -> Callable[[float], float]:
    """Return a simple ``t -> discount factor`` callable."""
    return lambda t: float(curve.discount(t))
