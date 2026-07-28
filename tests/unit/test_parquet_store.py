"""Tests for Parquet market data storage."""

from pathlib import Path

import pandas as pd
import pytest

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import Symbol
from vip.persistence import ParquetMarketDataStore


def test_parquet_roundtrip(tmp_path: Path) -> None:
    """Saving then loading returns an equivalent frame."""
    store = ParquetMarketDataStore(tmp_path)
    symbol = Symbol("SPY")
    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100, 200],
        }
    )

    assert not store.exists(symbol)
    written = store.save(symbol, frame)
    assert written.is_file()
    assert store.exists(symbol)

    loaded = store.load(symbol)
    pd.testing.assert_frame_equal(loaded, frame)


def test_load_missing_raises(tmp_path: Path) -> None:
    """Loading a missing symbol raises PersistenceError."""
    store = ParquetMarketDataStore(tmp_path)
    with pytest.raises(PersistenceError):
        store.load(Symbol("SPY"))