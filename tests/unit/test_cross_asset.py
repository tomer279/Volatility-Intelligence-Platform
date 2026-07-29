"""Tests for VIX cross-asset feature alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.features.cross_asset import build_vix_features
from vip.features.pipeline import build_feature_matrix


def _ohlcv(index: pd.DatetimeIndex, start: float = 100.0) -> pd.DataFrame:
    close = pd.Series(
        np.linspace(start, start + len(index) - 1, len(index)),
        index=index,
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000.0, 3_000.0, len(index)),
        },
        index=index,
    )


def test_vix_level_matches_same_day_close() -> None:
    """On shared sessions, vix_level must equal that day's VIX close."""
    index = pd.bdate_range("2024-01-02", periods=20)
    vix = _ohlcv(index, start=15.0)
    features = build_vix_features(index, vix)
    aligned = features.dropna()
    pd.testing.assert_series_equal(
        aligned["vix_level"],
        vix.loc[aligned.index, "close"],
        check_names=False,
    )


def test_pipeline_with_vix_adds_columns() -> None:
    """Feature matrix should include VIX columns when aux data is provided."""
    index = pd.bdate_range("2024-01-02", periods=60)
    primary = _ohlcv(index, start=100.0)
    vix = _ohlcv(index, start=15.0)
    matrix = build_feature_matrix(primary, horizon_days=5, vix_ohlcv=vix)
    assert "vix_level" in matrix.columns
    assert "vix_chg_1d" in matrix.columns
    assert not matrix.isna().any().any()