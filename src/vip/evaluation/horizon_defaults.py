"""Horizon-scaled embargo and bootstrap-block defaults for multi-horizon screens.

Exports
-------
LOCKED_SCREEN_HORIZONS
    Locked study horizons ``(1, 5, 21)``.
default_embargo_for_horizon
    Embargo size locked to ``horizon_days``.
default_bootstrap_block_length
    Locked default moving-block length for a horizon.
allowed_bootstrap_block_range
    Inclusive ``(min, max)`` block-length interval for a horizon.
validate_bootstrap_block_length
    Raise ``DataValidationError`` when ``block_length`` is outside the range.
"""

from __future__ import annotations

from vip.domain.errors import DataValidationError

LOCKED_SCREEN_HORIZONS: tuple[int, ...] = (1, 5, 21)

# Inclusive allowed ranges keyed by locked horizon (trading days).
_BLOCK_LENGTH_RANGE_BY_HORIZON: dict[int, tuple[int, int]] = {
    1: (5, 15),
    5: (10, 20),
    21: (15, 42),
}

# Locked defaults keyed by horizon (must lie inside the ranges above).
_DEFAULT_BLOCK_LENGTH_BY_HORIZON: dict[int, int] = {
    1: 10,
    5: 15,
    21: 21,
}

_MIN_HORIZON_DAYS = 1
_RANGE_LOW_INDEX = 0
_RANGE_HIGH_INDEX = 1


def _require_locked_horizon(horizon_days: int) -> None:
    """Raise when ``horizon_days`` is outside the locked study set.

    Parameters
    ----------
    horizon_days : int
        Forecast horizon in trading days.

    Raises
    ------
    DataValidationError
        If ``horizon_days`` is not in ``LOCKED_SCREEN_HORIZONS``.
    """
    if horizon_days not in LOCKED_SCREEN_HORIZONS:
        raise DataValidationError(
            f"horizon_days must be one of {LOCKED_SCREEN_HORIZONS}; "
            f"got {horizon_days}."
        )


def default_embargo_for_horizon(horizon_days: int) -> int:
    """Return the locked embargo size for a forecast horizon.

    Embargo equals the horizon so overlapping *h*-step labels cannot leak
    into the training window (``embargo_size = horizon_days``).

    Parameters
    ----------
    horizon_days : int
        Forecast horizon in trading days (must be >= 1).

    Returns
    -------
    int
        Embargo size in trading days (identical to ``horizon_days``).

    Raises
    ------
    DataValidationError
        If ``horizon_days`` is less than 1.
    """
    if horizon_days < _MIN_HORIZON_DAYS:
        raise DataValidationError("horizon_days must be at least 1.")
    return horizon_days


def default_bootstrap_block_length(horizon_days: int) -> int:
    """Return the locked default bootstrap block length for a horizon.

    Parameters
    ----------
    horizon_days : int
        Forecast horizon in trading days (must be in ``LOCKED_SCREEN_HORIZONS``).

    Returns
    -------
    int
        Default moving-block length (10 / 15 / 21 for h = 1 / 5 / 21).

    Raises
    ------
    DataValidationError
        If ``horizon_days`` is not a locked screen horizon.
    """
    _require_locked_horizon(horizon_days)
    return _DEFAULT_BLOCK_LENGTH_BY_HORIZON[horizon_days]


def allowed_bootstrap_block_range(horizon_days: int) -> tuple[int, int]:
    """Return the inclusive allowed block-length range for a horizon.

    Parameters
    ----------
    horizon_days : int
        Forecast horizon in trading days (must be in ``LOCKED_SCREEN_HORIZONS``).

    Returns
    -------
    tuple of int
        ``(minimum, maximum)`` inclusive bounds.

    Raises
    ------
    DataValidationError
        If ``horizon_days`` is not a locked screen horizon.
    """
    _require_locked_horizon(horizon_days)
    return _BLOCK_LENGTH_RANGE_BY_HORIZON[horizon_days]


def validate_bootstrap_block_length(
        horizon_days: int,
        block_length: int,
) -> None:
    """Validate ``block_length`` against the horizon-specific allowed range.

    Parameters
    ----------
    horizon_days : int
        Forecast horizon in trading days (must be in ``LOCKED_SCREEN_HORIZONS``).
    block_length : int
        Proposed moving-block length in trading days.

    Raises
    ------
    DataValidationError
        If the horizon is not locked or ``block_length`` is outside the
        inclusive allowed range for that horizon.
    """
    low, high = allowed_bootstrap_block_range(horizon_days)
    if block_length < low or block_length > high:
        raise DataValidationError(
            f"block_length must be in [{low}, {high}] "
            f"for horizon_days={horizon_days}; got {block_length}."
        )
