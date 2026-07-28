"""Tests for forward realized-volatility targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.features.targets import (
    build_target_rv_cc,
    daily_log_returns,
    realized_variance_forward,
    realized_volatility_forward,
)


def _toy_ohlcv() -> pd.DataFrame:
    """Build a tiny deterministic OHLCV frame for hand checks.

    Returns
    -------
    pandas.DataFrame
        Seven sessions with simple increasing closes.
    """
    index = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0], index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1_000] * len(close),
        },
        index=index,
    )


def test_daily_log_returns_matches_manual_values() -> None:
    """Daily log returns should match log close ratios."""
    close = _toy_ohlcv()["close"]
    returns = daily_log_returns(close)

    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(np.log(101.0 / 100.0))
    assert returns.iloc[2] == pytest.approx(np.log(102.0 / 101.0))


def test_forward_variance_uses_future_window_only() -> None:
    """Forward variance at t should sum r_{t+1}^2 .. r_{t+h}^2."""
    returns = daily_log_returns(_toy_ohlcv()["close"])
    horizon = 2
    variance = realized_variance_forward(returns, horizon)

    expected_first = returns.iloc[1] ** 2 + returns.iloc[2] ** 2
    assert variance.iloc[0] == pytest.approx(expected_first)

    expected_second = returns.iloc[2] ** 2 + returns.iloc[3] ** 2
    assert variance.iloc[1] == pytest.approx(expected_second)


def test_forward_volatility_is_sqrt_of_variance() -> None:
    """Forward volatility should be the square root of forward variance."""
    returns = daily_log_returns(_toy_ohlcv()["close"])
    horizon = 2
    variance = realized_variance_forward(returns, horizon)
    volatility = realized_volatility_forward(returns, horizon)

    assert volatility.iloc[0] == pytest.approx(np.sqrt(variance.iloc[0]))


def test_last_horizon_targets_are_nan() -> None:
    """The final horizon rows cannot observe a full future window."""
    returns = daily_log_returns(_toy_ohlcv()["close"])
    horizon = 2
    variance = realized_variance_forward(returns, horizon)

    assert variance.isna().iloc[-horizon:].all()
    assert variance.notna().iloc[:-horizon].all() or variance.iloc[0:1].notna().any()


def test_build_target_rv_cc_name_and_values() -> None:
    """build_target_rv_cc should name and compute the primary target."""
    frame = _toy_ohlcv()
    target = build_target_rv_cc(frame, horizon_days=2)

    assert target.name == "target_rv_cc_2d"
    returns = daily_log_returns(frame["close"])
    expected = np.sqrt(returns.iloc[1] ** 2 + returns.iloc[2] ** 2)
    assert target.iloc[0] == pytest.approx(expected)
    assert target.isna().iloc[-2:].all()


def test_invalid_horizon_raises() -> None:
    """Horizons below 1 should raise DataValidationError."""
    returns = daily_log_returns(_toy_ohlcv()["close"])
    with pytest.raises(DataValidationError, match="Horizon must be at least 1"):
        realized_variance_forward(returns, 0)
    with pytest.raises(DataValidationError, match="Horizon must be at least 1"):
        build_target_rv_cc(_toy_ohlcv(), horizon_days=0)


def test_missing_close_column_raises() -> None:
    """Target builder should require a close column."""
    frame = _toy_ohlcv().drop(columns=["close"])
    with pytest.raises(DataValidationError, match="close"):
        build_target_rv_cc(frame, horizon_days=2)