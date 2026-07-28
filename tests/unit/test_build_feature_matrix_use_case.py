"""Tests for feature-matrix application use-case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vip.application.build_feature_matrix import build_and_persist_feature_matrix
from vip.domain.value_objects import Symbol
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore


def _synthetic_ohlcv(n_rows: int = 80) -> pd.DataFrame:
    """Build synthetic canonical OHLCV data."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    close = pd.Series(np.linspace(100.0, 140.0, n_rows), index=index)
    return pd.DataFrame(
        {
            "open": close.to_numpy(),
            "high": (close + 1.0).to_numpy(),
            "low": (close - 1.0).to_numpy(),
            "close": close.to_numpy(),
            "volume": np.linspace(1_000.0, 3_000.0, n_rows),
        },
        index=index,
    )


def test_build_and_persist_feature_matrix(tmp_path: Path) -> None:
    """Use-case should load OHLCV, build matrix, and persist features."""
    market_root = tmp_path / "raw"
    feature_root = tmp_path / "processed"
    market_store = ParquetMarketDataStore(market_root)
    feature_store = ParquetFeatureMatrixStore(feature_root)

    symbol = Symbol("SPY")
    market_store.save(symbol, _synthetic_ohlcv())

    result = build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=symbol,
        horizon_days=5,
    )

    assert result.symbol == symbol
    assert result.row_count > 0
    assert result.feature_count == 8
    assert result.output_path.is_file()
    assert feature_store.exists(symbol)

    loaded = feature_store.load(symbol)
    assert "target_rv_cc_5d" in loaded.columns
    assert not loaded.isna().any().any()