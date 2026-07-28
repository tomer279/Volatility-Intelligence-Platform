"""Tests for trailing realized-volatility helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.features.realized import (
    realized_variance_trailing,
    realized_volatility_trailing,
)
from vip.features.targets import daily_log_returns


def _toy_returns() -> pd.Series:
    """Build a small deterministic return series.

    Returns
    -------
    pandas.Series
        Log returns from simple increasing closes.
    """
    index = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)
    return daily_log_returns(close)


def test_trailing_variance_matches_manual_window() -> None:
    """Trailing variance at t should sum the last ``window`` squared returns."""
    returns = _toy_returns()
    window = 3
    variance = realized_variance_trailing(returns, window)

    # First two rows incomplete (and first return is NaN anyway).
    assert variance.isna().iloc[: window].all() or variance.iloc[window - 1 : window].isna().any()

    end_idx = 4
    expected = float(
        returns.iloc[end_idx - window + 1 : end_idx + 1].pow(2).sum()
    )
    assert variance.iloc[end_idx] == pytest.approx(expected)


def test_trailing_volatility_is_sqrt_of_variance() -> None:
    """Trailing volatility should be sqrt of trailing variance."""
    returns = _toy_returns()
    window = 3
    variance = realized_variance_trailing(returns, window)
    volatility = realized_volatility_trailing(returns, window)

    valid = variance.notna()
    assert np.allclose(
        volatility[valid].to_numpy(),
        np.sqrt(variance[valid].to_numpy()),
    )


def test_trailing_uses_only_past_and_current_returns() -> None:
    """Truncating the future should not change trailing values at t."""
    returns = _toy_returns()
    window = 3
    full = realized_volatility_trailing(returns, window)

    cutoff = 4
    truncated = realized_volatility_trailing(returns.iloc[: cutoff + 1], window)
    assert full.iloc[cutoff] == pytest.approx(truncated.iloc[cutoff])


def test_invalid_window_raises() -> None:
    """Windows below 1 should raise DataValidationError."""
    returns = _toy_returns()
    with pytest.raises(DataValidationError, match="Window must be at least 1"):
        realized_variance_trailing(returns, 0)
