"""Shared fixtures for integration tests.

Fixtures
--------
mock_source
    Deterministic MarketDataSource that requires no network access.
market_store
    ParquetMarketDataStore backed by a temporary directory.
feature_store
    ParquetFeatureMatrixStore backed by a temporary directory.
artifact_store
    FilesystemArtifactStore backed by a temporary directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vip.domain.value_objects import DateRange, Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore

N_ROWS = 252
BASE_PRICE = 400.0
DAILY_RETURN_STD = 0.012
DAILY_RETURN_MEAN = 0.0005
OPEN_NOISE_STD = 0.002
RANGE_NOISE_STD = 0.005
VOLUME_LOW = 3e7
VOLUME_HIGH = 8e7


@dataclass(frozen=True, slots=True)
class MockMarketDataSource:
    """Deterministic synthetic market data source for integration tests.

    Generates realistic SPY-like OHLCV from a fixed random seed so that
    integration tests are fully network-free and reproducible.

    Parameters
    ----------
    seed : int, default 42
        NumPy random seed for reproducibility.
    n_rows : int, default 252
        Number of trading days to generate.

    Methods
    -------
    fetch(symbol, date_range)
        Return synthetic OHLCV for the requested symbol.
    source_name()
        Return the stable source identifier.
    """

    seed: int = 42
    n_rows: int = N_ROWS

    def fetch(self, symbol: Symbol, date_range: DateRange) -> pd.DataFrame:
        """Return deterministic synthetic OHLCV data.

        Parameters
        ----------
        symbol : Symbol
            Requested instrument (ignored; same synthetic data for all).
        date_range : DateRange
            Requested window (start date used for index generation).

        Returns
        -------
        pandas.DataFrame
            Canonical OHLCV with ``open, high, low, close, volume`` columns
            and a business-day ``DatetimeIndex``.
        """
        rng = np.random.RandomState(self.seed)
        index = pd.bdate_range(date_range.start, periods=self.n_rows)

        returns = rng.normal(DAILY_RETURN_MEAN, DAILY_RETURN_STD, self.n_rows)
        close = BASE_PRICE * np.cumprod(1.0 + returns)

        open_noise = rng.normal(0.0, OPEN_NOISE_STD, self.n_rows)
        open_ = close * (1.0 + open_noise)

        high_spread = np.abs(rng.normal(0.0, RANGE_NOISE_STD, self.n_rows))
        low_spread = np.abs(rng.normal(0.0, RANGE_NOISE_STD, self.n_rows))
        high = np.maximum(open_, close) * (1.0 + high_spread)
        low = np.minimum(open_, close) * (1.0 - low_spread)

        volume = rng.uniform(VOLUME_LOW, VOLUME_HIGH, self.n_rows).astype(int)

        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )

    def source_name(self) -> str:
        """Return the stable source identifier.

        Returns
        -------
        str
            Always ``"mock"``.
        """
        return "mock"


@pytest.fixture()
def mock_source() -> MockMarketDataSource:
    """Deterministic mock market data source (seed=42, 252 rows)."""
    return MockMarketDataSource()


@pytest.fixture()
def market_store(tmp_path: Path) -> ParquetMarketDataStore:
    """Parquet market-data store in a temporary directory."""
    return ParquetMarketDataStore(root_dir=tmp_path / "raw")


@pytest.fixture()
def feature_store(tmp_path: Path) -> ParquetFeatureMatrixStore:
    """Parquet feature-matrix store in a temporary directory."""
    return ParquetFeatureMatrixStore(root_dir=tmp_path / "processed")


@pytest.fixture()
def artifact_store(tmp_path: Path) -> FilesystemArtifactStore:
    """Filesystem artifact store in a temporary directory."""
    return FilesystemArtifactStore(root_dir=tmp_path / "artifacts")
