"""Live Dataloader smoke tests, skipped when the API keys are not set."""

from __future__ import annotations

import pandas as pd
import pytest

from data.data_loader import load_ohlcv, wide_close


@pytest.mark.market_data
def test_dataloader_smoke_small_window():
    bars, meta = load_ohlcv(["AAPL", "MSFT"], "2024-01-02", "2024-01-16",
                            timeframe="1d")
    assert not bars.empty
    for col in ("timestamp", "date", "ticker", "open", "high", "low", "close", "volume"):
        assert col in bars.columns
    assert set(bars["ticker"].unique()) == {"AAPL", "MSFT"}
    assert meta.get("ticker_details"), "ticker_details must be populated"


@pytest.mark.market_data
def test_wide_close_pivot():
    bars, _ = load_ohlcv(["AAPL", "MSFT"], "2024-01-02", "2024-01-31",
                         timeframe="1d")
    wide = wide_close(bars, ["AAPL", "MSFT"])
    assert isinstance(wide.index, pd.DatetimeIndex)
    assert list(wide.columns) == ["AAPL", "MSFT"]
    assert (wide > 0).all().all()
