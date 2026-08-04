"""Trailing realized-volatility helpers for feature construction.

Exports
-------
realized_variance_trailing
    Backward sum of squared returns ending at session ``t``.
realized_volatility_trailing
    Backward realized volatility ending at session ``t``.
bipower_variation_trailing
    Trailing daily bipower-variation proxy ending at session ``t``.
bipower_volatility_trailing
    Square root of trailing bipower variation.
jump_proportion_trailing
    Trailing jump-proportion proxy ``max(0, RV - BPV) / RV``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vip.domain.errors import DataValidationError



# μ₁ = E[|Z|] for Z~N(0,1); bipower scale is μ₁^{-2} = π/2.
BIPOWER_SCALE = np.pi / 2.0
WINDOW_UNIT = 1
PAIR_COUNT_OFFSET = 1


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


def bipower_variation_trailing(returns: pd.Series, window: int) -> pd.Series:
    """Compute trailing daily bipower variation ending at each session ``t``.

    Uses the close-to-close proxy
    ``(π/2) * sum_i |r_i| |r_{i-1}|`` over adjacent pairs inside the
    trailing window. For ``window >= 2`` there are ``window - 1`` pairs
    among the last ``window`` returns. For ``window == 1`` a single pair
    ``|r_t| |r_{t-1}|`` is used so the label aligns with HAR's 1d lag.
    This is **not** high-frequency / tick bipower variation.

    Parameters
    ----------
    returns : pandas.Series
        Daily log returns indexed by session date.
    window : int
        Trailing window length in trading days. Must be >= 1.

    Returns
    -------
    pandas.Series
        Trailing bipower variation. Early incomplete rows are NaN.

    Raises
    ------
    DataValidationError
        If ``window`` is less than 1.
    """
    _validate_window(window)
    abs_returns = returns.abs()
    adjacent_product = abs_returns * abs_returns.shift(WINDOW_UNIT)
    pair_count = _bipower_pair_count(window)
    variation = (
        BIPOWER_SCALE
        * adjacent_product.rolling(window=pair_count, min_periods=pair_count).sum()
    )
    variation.name = f"bpv2_trail_{window}d"
    return variation


def bipower_volatility_trailing(returns: pd.Series, window: int) -> pd.Series:
    """Compute trailing daily bipower volatility ending at session ``t``.

    Parameters
    ----------
    returns : pandas.Series
        Daily log returns indexed by session date.
    window : int
        Trailing window length in trading days. Must be >= 1.

    Returns
    -------
    pandas.Series
        ``sqrt`` of trailing bipower variation. Non-annualized.

    Raises
    ------
    DataValidationError
        If ``window`` is less than 1.
    """
    variation = bipower_variation_trailing(returns, window)
    volatility = np.sqrt(variation)
    volatility.name = f"bpv_trail_{window}d"
    return volatility


def jump_proportion_trailing(returns: pd.Series, window: int) -> pd.Series:
    """Compute trailing jump proportion ``max(0, RV - BPV) / RV``.
    When realized variance is exactly zero, the proportion is set to
    ``0.0``. Incomplete windows remain NaN.

    Parameters
    ----------
    returns : pandas.Series
        Daily log returns indexed by session date.
    window : int
        Trailing window length in trading days. Must be >= 1.

    Returns
    -------
    pandas.Series
        Jump-proportion proxy in ``[0, 1]`` where defined.

    Raises
    ------
    DataValidationError
        If ``window`` is less than 1.
    """
    realized_var = realized_variance_trailing(returns, window)
    bipower_var = bipower_variation_trailing(returns, window)
    positive_gap = (realized_var - bipower_var).clip(lower=0.0)
    proportion = pd.Series(np.nan, index=returns.index, dtype=float)
    positive_rv = realized_var > 0.0
    zero_rv = realized_var == 0.0
    proportion.loc[positive_rv] = (
        positive_gap.loc[positive_rv] / realized_var.loc[positive_rv]
    )
    proportion.loc[zero_rv] = 0.0
    proportion.name = f"jump_prop_trail_{window}d"
    return proportion


def _bipower_pair_count(window: int) -> int:
    """Return the number of adjacent absolute-return products for ``window``.

    Parameters
    ----------
    window : int
        Trailing RV window length (already validated >= 1).

    Returns
    -------
    int
        Rolling length applied to ``|r_i| |r_{i-1}|`` products.
    """
    if window == WINDOW_UNIT:
        return WINDOW_UNIT
    return window - PAIR_COUNT_OFFSET
