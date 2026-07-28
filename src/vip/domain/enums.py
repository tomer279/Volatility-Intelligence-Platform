"""Enumerations used across the VIP domain.

Exports
-------
RvEstimator
    Realized-volatility estimators.
MetricName
    Forecast evaluation metrics.
PriceFrequency
    Market data sampling frequency.
SplitMode
    Temporal validation strategies.
"""

from enum import StrEnum


class RvEstimator(StrEnum):
    """Supported realized-volatility estimators."""

    CLOSE_TO_CLOSE = "close_to_close"


class MetricName(StrEnum):
    """Supported forecast evaluation metrics."""

    QLIKE = "qlike"
    MSE = "mse"
    MAE = "mae"


class PriceFrequency(StrEnum):
    """Market data bar frequency."""

    DAILY = "daily"


class SplitMode(StrEnum):
    """Temporal train/test split strategies."""

    WALK_FORWARD = "walk_forward"