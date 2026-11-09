"""Unit tests for the yield curve construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rates import YieldCurve, build_curve, flat_curve


def test_flat_curve_discount():
    c = flat_curve(0.05, as_of=pd.Timestamp("2024-01-01"))
    assert c.discount(1.0) == pytest.approx(np.exp(-0.05))
    assert c.discount(0.0) == pytest.approx(1.0)
    assert c.forward(0.5, 1.0) == pytest.approx(0.05)


def test_build_curve_rejects_unknown():
    with pytest.raises(ValueError):
        build_curve("quadratic", flat_rate=0.01)


def test_yield_curve_zero_requires_nonneg_time():
    c = flat_curve(0.03)
    with pytest.raises(ValueError):
        c.zero(-1.0)
