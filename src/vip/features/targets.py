"""Forward realized-volatility target construction.

Exports
-------
daily_log_returns
    Daily log returns from close prices.
realized_variance_forward
    Forward sum of squared returns over a horizon.
realized_volatility_forward
    Forward realized volatility over a horizon.
build_target_rv_cc
    Build a named close-to-close forward RV target from OHLCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vip.domain.errors import DataValidationError

CLOSE_COLUMN = "close"
TARGET_NAME_PREFIX = "target_rv_cc_"


def daily_log_returns(close: pd.Series) -> pd.Series:
    """Compute daily log returns from a close-price series.

    Parameters
    ----------
    close : pandas.Series
        Close prices indexed by session date, ordered by trading day.

    Returns
    -------
    pandas.Series
        Log returns ``log(close_t / close_{t-1})``. The first value is NaN.

    Raises
    ------
    DataValidationError
        If ``close`` is empty.
    """
    if close.empty:
        raise DataValidationError("Close series must be non-empty.")
    return np.log(close / close.shift(1))


def realized_variance_forward(returns: pd.Series, horizon: int) -> pd.Series:
    """Compute forward realized variance over the next ``horizon`` sessions.

    For each session ``t``, the value is the sum of squared returns
    ``r_{t+1}^2 + ... + r_{t+horizon}^2``.

    Parameters
    ----------
    returns : pandas.Series
        Daily log returns indexed by session date.
    horizon : int
        Number of future trading days in the window. Must be >= 1.

    Returns
    -------
    pandas.Series
        Forward realized variance aligned to ``t``. The last ``horizon``
        values are NaN because the future window is incomplete.

    Raises
    ------
    DataValidationError
        If ``horizon`` is less than 1.
    """
    _validate_horizon(horizon)
    squared = returns.to_numpy(dtype=float) ** 2
    n_rows = squared.shape[0]
    values = np.full(n_rows, np.nan, dtype=float)

    for index in range(n_rows - horizon):
        window = squared[index + 1 : index + 1 + horizon]
        values[index] = float(np.sum(window))

    return pd.Series(values, index=returns.index, name=f"rv2_fwd_{horizon}d")


def realized_volatility_forward(returns: pd.Series, horizon: int) -> pd.Series:
    """Compute forward realized volatility over the next ``horizon`` sessions.

    Parameters
    ----------
    returns : pandas.Series
        Daily log returns indexed by session date.
    horizon : int
        Number of future trading days in the window. Must be >= 1.

    Returns
    -------
    pandas.Series
        ``sqrt`` of forward realized variance. Non-annualized.

    Raises
    ------
    DataValidationError
        If ``horizon`` is less than 1.
    """
    variance = realized_variance_forward(returns, horizon)
    volatility = np.sqrt(variance)
    volatility.name = f"rv_fwd_{horizon}d"
    return volatility


def build_target_rv_cc(
    ohlcv: pd.DataFrame,
    horizon_days: int = 5,
) -> pd.Series:
    """Build a forward close-to-close realized-volatility target.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Canonical OHLCV frame containing a ``close`` column.
    horizon_days : int, default 5
        Forecast horizon in trading days.

    Returns
    -------
    pandas.Series
        Target series named ``target_rv_cc_{horizon_days}d``.

    Raises
    ------
    DataValidationError
        If ``close`` is missing or ``horizon_days`` is invalid.
    """
    _validate_horizon(horizon_days)
    if CLOSE_COLUMN not in ohlcv.columns:
        raise DataValidationError("OHLCV frame must contain a 'close' column.")

    returns = daily_log_returns(ohlcv[CLOSE_COLUMN])
    target = realized_volatility_forward(returns, horizon_days)
    target.name = f"{TARGET_NAME_PREFIX}{horizon_days}d"
    return target


def _validate_horizon(horizon: int) -> None:
    """Validate that a horizon is a positive integer.

    Parameters
    ----------
    horizon : int
        Candidate horizon in trading days.

    Raises
    ------
    DataValidationError
        If ``horizon`` is less than 1.
    """
    if horizon < 1:
        raise DataValidationError("Horizon must be at least 1 trading day.")
