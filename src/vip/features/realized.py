"""Trailing realized-volatility helpers for feature construction.

Exports
-------
realized_variance_trailing
    Backward sum of squared returns ending at session ``t``.
realized_volatility_trailing
    Backward realized volatility ending at session ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vip.domain.errors import DataValidationError


def realized_variance_trailing(returns: pd.Series, window: int) -> pd.Series:
    """Compute trailing realized variance ending at each session ``t``.

    For each session ``t``, the value is the sum of squared returns
    ``r_{t-window+1}^2 + ... + r_t^2`` when the full window is available.

    Parameters
    ----------
    returns : pandas.Series
        Daily log returns indexed by session date.
    window : int
        Number of trading days in the trailing window. Must be >= 1.

    Returns
    -------
    pandas.Series
        Trailing realized variance. Early rows with an incomplete window
        are NaN.

    Raises
    ------
    DataValidationError
        If ``window`` is less than 1.
    """
    _validate_window(window)
    squared = returns ** 2
    trailing = squared.rolling(window=window, min_periods=window).sum()
    trailing.name = f"rv2_trail_{window}d"
    return trailing


def realized_volatility_trailing(returns: pd.Series, window: int) -> pd.Series:
    """Compute trailing realized volatility ending at each session ``t``.

    Parameters
    ----------
    returns : pandas.Series
        Daily log returns indexed by session date.
    window : int
        Number of trading days in the trailing window. Must be >= 1.

    Returns
    -------
    pandas.Series
        ``sqrt`` of trailing realized variance. Non-annualized.

    Raises
    ------
    DataValidationError
        If ``window`` is less than 1.
    """
    variance = realized_variance_trailing(returns, window)
    volatility = np.sqrt(variance)
    volatility.name = f"rv_trail_{window}d"
    return volatility


def _validate_window(window: int) -> None:
    """Validate that a trailing window is a positive integer.

    Parameters
    ----------
    window : int
        Candidate window length in trading days.

    Raises
    ------
    DataValidationError
        If ``window`` is less than 1.
    """
    if window < 1:
        raise DataValidationError("Window must be at least 1 trading day.")
