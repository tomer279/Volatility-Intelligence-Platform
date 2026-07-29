"""Leakage tests for VIX as-of alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.features.cross_asset import build_vix_features


def _vix_ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(np.linspace(10.0, 30.0, len(index)), index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(index), 0.0),
        },
        index=index,
    )


def test_vix_feature_at_t_ignores_future_vix_shock() -> None:
    """Shocking VIX after t must not change features at t."""
    index = pd.bdate_range("2024-01-02", periods=30)
    base = _vix_ohlcv(index)
    t_date = index[10]
    shock_date = index[12]

    shocked = base.copy()
    shocked.loc[shock_date, "close"] = shocked.loc[shock_date, "close"] * 2.0
    shocked.loc[shock_date, "open"] = shocked.loc[shock_date, "close"]
    shocked.loc[shock_date, "high"] = shocked.loc[shock_date, "close"] + 1.0
    shocked.loc[shock_date, "low"] = shocked.loc[shock_date, "close"] - 1.0

    base_feat = build_vix_features(index, base)
    shocked_feat = build_vix_features(index, shocked)
    pd.testing.assert_series_equal(
        base_feat.loc[t_date],
        shocked_feat.loc[t_date],
        check_names=False,
    )


def test_vix_feature_at_t_is_not_next_day_close() -> None:
    """vix_level at t must equal close_t, not close_{t+1}."""
    index = pd.bdate_range("2024-01-02", periods=15)
    vix = _vix_ohlcv(index)
    features = build_vix_features(index, vix)
    t_date = index[5]
    next_date = index[6]
    assert features.loc[t_date, "vix_level"] == pytest.approx(vix.loc[t_date, "close"])
    assert features.loc[t_date, "vix_level"] != pytest.approx(
        vix.loc[next_date, "close"]
    )