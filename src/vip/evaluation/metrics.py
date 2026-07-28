"""Scalar metrics for realized-volatility forecasts.

Exports
-------
mse
    Mean squared error.
mae
    Mean absolute error.
qlike
    QLIKE volatility forecast loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vip.domain.errors import DataValidationError

DEFAULT_EPSILON = 1e-8


def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute mean squared error.

    Parameters
    ----------
    y_true : pandas.Series
        Realized target values.
    y_pred : pandas.Series
        Predicted values.

    Returns
    -------
    float
        Mean of squared residuals.

    Raises
    ------
    DataValidationError
        If inputs are empty or cannot be aligned.
    """
    left, right = _align_and_validate(y_true, y_pred)
    return float(np.mean((left - right) ** 2))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute mean absolute error.

    Parameters
    ----------
    y_true : pandas.Series
        Realized target values.
    y_pred : pandas.Series
        Predicted values.

    Returns
    -------
    float
        Mean of absolute residuals.

    Raises
    ------
    DataValidationError
        If inputs are empty or cannot be aligned.
    """
    left, right = _align_and_validate(y_true, y_pred)
    return float(np.mean(np.abs(left - right)))


def qlike(
    y_true: pd.Series,
    y_pred: pd.Series,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Compute QLIKE loss for volatility forecasts.

    Uses ``mean(log(yhat^2) + y^2 / yhat^2)`` with predictions clipped
    below by ``epsilon`` to avoid non-positive forecasts.

    Parameters
    ----------
    y_true : pandas.Series
        Realized volatility values.
    y_pred : pandas.Series
        Forecasted volatility values.
    epsilon : float, default 1e-8
        Lower bound applied to predictions before scoring.

    Returns
    -------
    float
        QLIKE loss (lower is better).

    Raises
    ------
    DataValidationError
        If inputs are empty, cannot be aligned, or ``epsilon`` is not positive.
    """
    if epsilon <= 0:
        raise DataValidationError("epsilon must be positive.")

    left, right = _align_and_validate(y_true, y_pred)
    clipped = np.maximum(right.to_numpy(dtype=float), epsilon)
    realized = left.to_numpy(dtype=float)
    values = np.log(clipped**2) + (realized**2) / (clipped**2)
    return float(np.mean(values))


def _align_and_validate(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align series on a common index and drop missing pairs.

    Parameters
    ----------
    y_true : pandas.Series
        Realized target values.
    y_pred : pandas.Series
        Predicted values.

    Returns
    -------
    tuple of pandas.Series
        Aligned ``(y_true, y_pred)`` with no NaNs.

    Raises
    ------
    DataValidationError
        If no overlapping finite observations remain.
    """
    aligned = pd.concat(
        [y_true.rename("y_true"), y_pred.rename("y_pred")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise DataValidationError(
            "No overlapping finite observations available for metric computation."
        )
    return aligned["y_true"], aligned["y_pred"]
