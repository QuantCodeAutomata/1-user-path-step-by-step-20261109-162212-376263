"""Shared pytest fixtures.

* Registers the ``market_data`` marker so live-API tests are explicit.
* Skips those tests when the Dataloader / Massive credentials are missing.
* Provides a session-scoped synthetic price frame for unit tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "market_data: integration test that calls the QuantX Dataloader "
        "and Massive APIs; skipped when credentials are missing.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    load_dotenv()
    if os.environ.get("DATALOADER_API_KEY") and os.environ.get("MASSIVE_API_KEY"):
        return
    skip_live = pytest.mark.skip(reason="DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    for item in items:
        if "market_data" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    """Reproducible synthetic basket, 2 years of daily bars, 5 assets."""
    from data.generate_synthetic import synthetic_prices as _gen
    return _gen(
        tickers=["A", "B", "C", "D", "E"],
        start="2022-01-03",
        end="2023-12-29",
        spot=np.array([100.0, 50.0, 200.0, 80.0, 120.0]),
        vol=np.array([0.25, 0.30, 0.20, 0.22, 0.28]),
        drift=np.array([0.05, 0.03, 0.07, 0.04, 0.06]),
        seed=42,
    )
