"""Tests for Milestone 2 feature family builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.features.har import build_har_features
from vip.features.range_features import build_range_features
from vip.features.returns import build_return_features
from vip.features.targets import daily_log_returns
from vip.features.volume_features import build_volume_features


def _synthetic_ohlcv(n_rows: int = 40) -> pd.DataFrame:
    """Build a synthetic canonical OHLCV frame.

    Parameters
    ----------
    n_rows : int, default 40
        Number of trading sessions.

    Returns
    -------
    pandas.DataFrame
        Synthetic OHLCV data with a datetime index.
    """
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    close = pd.Series(np.linspace(100.0, 120.0, n_rows), index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, n_rows),
        },
        index=index,
    )


def test_return_features_columns_and_values() -> None:
    """Return features should match log close ratios."""
    frame = _synthetic_ohlcv()
    features = build_return_features(frame)

    assert list(features.columns) == ["ret_1d", "ret_5d"]
    assert features["ret_1d"].iloc[1] == pytest.approx(
        np.log(frame["close"].iloc[1] / frame["close"].iloc[0])
    )
    assert features["ret_5d"].iloc[5] == pytest.approx(
        np.log(frame["close"].iloc[5] / frame["close"].iloc[0])
    )


def test_har_features_match_trailing_rv() -> None:
    """HAR features should equal trailing realized volatility windows."""
    frame = _synthetic_ohlcv()
    features = build_har_features(frame)
    returns = daily_log_returns(frame["close"])

    assert list(features.columns) == ["rv_cc_1d", "rv_cc_5d", "rv_cc_21d"]
    assert features["rv_cc_1d"].iloc[10] == pytest.approx(np.sqrt(returns.iloc[10] ** 2))
    expected_5 = float(np.sqrt(returns.iloc[6:11].pow(2).sum()))
    assert features["rv_cc_5d"].iloc[10] == pytest.approx(expected_5)


def test_range_features_values() -> None:
    """Range features should use high/low/close at t and trailing means."""
    frame = _synthetic_ohlcv()
    features = build_range_features(frame)

    expected_1d = (frame["high"] - frame["low"]) / frame["close"]
    assert features["range_1d"].iloc[10] == pytest.approx(expected_1d.iloc[10])
    expected_mean = float(expected_1d.iloc[6:11].mean())
    assert features["range_5d_mean"].iloc[10] == pytest.approx(expected_mean)


def test_volume_zscore_values() -> None:
    """Volume z-score should use trailing 21d mean and std."""
    frame = _synthetic_ohlcv(n_rows=40)
    features = build_volume_features(frame)

    volume = frame["volume"]
    mean_21 = volume.iloc[10:31].mean()  # window ending at index 30
    std_21 = volume.iloc[10:31].std(ddof=1)
    expected = (volume.iloc[30] - mean_21) / std_21
    assert features["volume_z_21d"].iloc[30] == pytest.approx(expected)