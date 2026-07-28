"""Tests for feature-matrix pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.features.pipeline import build_feature_matrix


def _synthetic_ohlcv(n_rows: int = 60) -> pd.DataFrame:
    """Build synthetic vendor-like OHLCV data."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    close = pd.Series(np.linspace(100.0, 130.0, n_rows), index=index)
    return pd.DataFrame(
        {
            "Open": close.to_numpy(),
            "High": (close + 1.0).to_numpy(),
            "Low": (close - 1.0).to_numpy(),
            "Close": close.to_numpy(),
            "Volume": np.linspace(1_000.0, 3_000.0, n_rows),
        },
        index=index,
    )


def test_build_feature_matrix_schema_and_no_nans() -> None:
    """Pipeline should return expected columns with no missing values."""
    matrix = build_feature_matrix(_synthetic_ohlcv(), horizon_days=5)

    expected_suffixes = [
        "ret_1d",
        "ret_5d",
        "rv_cc_1d",
        "rv_cc_5d",
        "rv_cc_21d",
        "range_1d",
        "range_5d_mean",
        "volume_z_21d",
        "target_rv_cc_5d",
    ]
    assert list(matrix.columns) == expected_suffixes
    assert not matrix.isna().any().any()
    assert matrix.index.is_monotonic_increasing


def test_build_feature_matrix_subset_families() -> None:
    """Pipeline should honor feature-family subset selection."""
    matrix = build_feature_matrix(
        _synthetic_ohlcv(),
        horizon_days=5,
        feature_names=["returns", "range"],
    )
    assert list(matrix.columns) == [
        "ret_1d",
        "ret_5d",
        "range_1d",
        "range_5d_mean",
        "target_rv_cc_5d",
    ]


def test_build_feature_matrix_too_short_raises() -> None:
    """Very short histories should fail after NaN dropping."""
    short = _synthetic_ohlcv(n_rows=10)
    with pytest.raises(DataValidationError, match="empty after dropping"):
        build_feature_matrix(short, horizon_days=5)


def test_invalid_horizon_raises() -> None:
    """Invalid horizons should raise DataValidationError."""
    with pytest.raises(DataValidationError, match="Horizon must be at least 1"):
        build_feature_matrix(_synthetic_ohlcv(), horizon_days=0)