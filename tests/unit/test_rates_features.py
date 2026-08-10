"""Rates (TNX) cross-asset builders and leakage tests.

test_rates_level_matches_same_day_close
    On shared sessions, tnx_level equals that day's close.
test_rates_feature_at_t_ignores_future_shock
    Shocking TNX after t must not change features at t.
test_rates_feature_at_t_is_not_next_day_close
    tnx_level at t equals close_t, not close_{t+1}.
test_pipeline_rates_only_adds_columns
    Rates join without VIX appends tnx_* columns.
test_pipeline_rates_and_vix_together
    Both families can coexist on one matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.features.cross_asset import (
    TNX_CHG_COLUMN,
    TNX_LEVEL_COLUMN,
    build_rates_features,
)
from vip.features.pipeline import VixJoinOptions, build_feature_matrix

T_INDEX = 10
SHOCK_OFFSET = 2
CHECK_INDEX = 5
NEXT_OFFSET = 1


def _ohlcv(index: pd.DatetimeIndex, start: float = 4.0) -> pd.DataFrame:
    close = pd.Series(
        np.linspace(start, start + 0.01 * (len(index) - 1), len(index)),
        index=index,
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.linspace(1_000.0, 3_000.0, len(index)),
        },
        index=index,
    )


def test_rates_level_matches_same_day_close() -> None:
    """On shared sessions, tnx_level must equal that day's TNX close."""
    index = pd.bdate_range("2024-01-02", periods=20)
    rates = _ohlcv(index, start=4.0)
    features = build_rates_features(index, rates)
    aligned = features.dropna(subset=[TNX_LEVEL_COLUMN])
    pd.testing.assert_series_equal(
        aligned[TNX_LEVEL_COLUMN],
        rates.loc[aligned.index, "close"],
        check_names=False,
    )


def test_rates_feature_at_t_ignores_future_shock() -> None:
    """Shocking TNX after t must not change features at t."""
    index = pd.bdate_range("2024-01-02", periods=30)
    base = _ohlcv(index)
    t_date = index[T_INDEX]
    shock_date = index[T_INDEX + SHOCK_OFFSET]

    shocked = base.copy()
    shocked.loc[shock_date, "close"] = shocked.loc[shock_date, "close"] * 2.0
    shocked.loc[shock_date, "open"] = shocked.loc[shock_date, "close"]
    shocked.loc[shock_date, "high"] = shocked.loc[shock_date, "close"] + 0.05
    shocked.loc[shock_date, "low"] = shocked.loc[shock_date, "close"] - 0.05

    base_feat = build_rates_features(index, base)
    shocked_feat = build_rates_features(index, shocked)
    pd.testing.assert_series_equal(
        base_feat.loc[t_date],
        shocked_feat.loc[t_date],
        check_names=False,
    )


def test_rates_feature_at_t_is_not_next_day_close() -> None:
    """tnx_level at t must equal close_t, not close_{t+1}."""
    index = pd.bdate_range("2024-01-02", periods=15)
    rates = _ohlcv(index)
    features = build_rates_features(index, rates)
    t_date = index[CHECK_INDEX]
    next_date = index[CHECK_INDEX + NEXT_OFFSET]
    assert features.loc[t_date, TNX_LEVEL_COLUMN] == pytest.approx(
        rates.loc[t_date, "close"]
    )
    assert features.loc[t_date, TNX_LEVEL_COLUMN] != pytest.approx(
        rates.loc[next_date, "close"]
    )


def test_pipeline_rates_only_adds_columns() -> None:
    """Feature matrix includes tnx columns when rates OHLCV is provided."""
    index = pd.bdate_range("2024-01-02", periods=60)
    primary = _ohlcv(index, start=100.0)
    rates = _ohlcv(index, start=4.0)
    matrix = build_feature_matrix(
        primary,
        horizon_days=5,
        vix_ohlcv=VixJoinOptions(rates_ohlcv=rates),
    )
    assert TNX_LEVEL_COLUMN in matrix.columns
    assert TNX_CHG_COLUMN in matrix.columns
    assert "vix_level" not in matrix.columns


def test_pipeline_rates_and_vix_together() -> None:
    """VIX and rates joins can both be present."""
    index = pd.bdate_range("2024-01-02", periods=60)
    primary = _ohlcv(index, start=100.0)
    vix = _ohlcv(index, start=15.0)
    rates = _ohlcv(index, start=4.0)
    matrix = build_feature_matrix(
        primary,
        horizon_days=5,
        vix_ohlcv=VixJoinOptions(vix_ohlcv=vix, rates_ohlcv=rates),
    )
    assert {"vix_level", TNX_LEVEL_COLUMN} <= set(matrix.columns)