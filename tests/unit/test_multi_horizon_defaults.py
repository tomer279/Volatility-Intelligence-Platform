"""Locked multi-horizon embargo / NW / bootstrap-block defaults."""

from __future__ import annotations

import pytest

from vip.domain.errors import DataValidationError
from vip.evaluation.horizon_defaults import (
    LOCKED_SCREEN_HORIZONS,
    allowed_bootstrap_block_range,
    default_bootstrap_block_length,
    default_embargo_for_horizon,
    validate_bootstrap_block_length,
)
from vip.evaluation.inference import (
    BootstrapBlockBounds,
    BootstrapInferenceOptions,
    nw_lags_for_horizon,
)

# Locked table: (horizon, embargo, nw_lags, default_block, range_low, range_high)
_LOCKED_ROWS: tuple[tuple[int, int, int, int, int, int], ...] = (
    (1, 1, 0, 10, 5, 15),
    (5, 5, 4, 15, 10, 20),
    (21, 21, 20, 21, 15, 42),
)
_HORIZON_INDEX = 0
_EMBARGO_INDEX = 1
_NW_LAGS_INDEX = 2
_DEFAULT_BLOCK_INDEX = 3
_RANGE_LOW_INDEX = 4
_RANGE_HIGH_INDEX = 5


@pytest.mark.parametrize("row", _LOCKED_ROWS)
def test_locked_horizon_defaults_table(row: tuple[int, int, int, int, int, int]) -> None:
    """Embargo, NW lags, default block length, and allowed range match the lock."""
    horizon = row[_HORIZON_INDEX]
    assert default_embargo_for_horizon(horizon) == row[_EMBARGO_INDEX]
    assert nw_lags_for_horizon(horizon) == row[_NW_LAGS_INDEX]
    assert default_bootstrap_block_length(horizon) == row[_DEFAULT_BLOCK_INDEX]
    assert allowed_bootstrap_block_range(horizon) == (
        row[_RANGE_LOW_INDEX],
        row[_RANGE_HIGH_INDEX],
    )
    validate_bootstrap_block_length(horizon, row[_DEFAULT_BLOCK_INDEX])


def test_locked_screen_horizons_constant() -> None:
    """Study set is exactly 1 / 5 / 21."""
    assert LOCKED_SCREEN_HORIZONS == (1, 5, 21)


def test_h21_default_block_is_legal_via_options() -> None:
    """BootstrapInferenceOptions must accept ℓ=21 when bounds are h=21."""
    low, high = allowed_bootstrap_block_range(21)
    options = BootstrapInferenceOptions(
        block_length=default_bootstrap_block_length(21),
        block_bounds=BootstrapBlockBounds(minimum=low, maximum=high),
    )
    options.validate()


def test_h21_default_block_rejected_under_legacy_bounds() -> None:
    """Legacy / default bounds remain [10, 20] so bare ℓ=21 still fails."""
    options = BootstrapInferenceOptions(block_length=21)
    with pytest.raises(DataValidationError, match="block_length must be in"):
        options.validate()


@pytest.mark.parametrize(
    ("horizon", "block_length"),
    [
        (1, 4),
        (1, 16),
        (5, 9),
        (5, 21),
        (21, 14),
        (21, 43),
    ],
)
def test_invalid_block_length_raises(horizon: int, block_length: int) -> None:
    """Lengths outside the horizon-specific inclusive range raise."""
    with pytest.raises(DataValidationError, match="block_length must be in"):
        validate_bootstrap_block_length(horizon, block_length)


def test_unknown_horizon_block_helpers_raise() -> None:
    """Block defaults/validation reject horizons outside the locked set."""
    with pytest.raises(DataValidationError, match="horizon_days must be one of"):
        default_bootstrap_block_length(7)
    with pytest.raises(DataValidationError, match="horizon_days must be one of"):
        allowed_bootstrap_block_range(7)
    with pytest.raises(DataValidationError, match="horizon_days must be one of"):
        validate_bootstrap_block_length(7, 15)


def test_embargo_rejects_non_positive_horizon() -> None:
    """Embargo helper requires horizon_days >= 1."""
    with pytest.raises(DataValidationError, match="horizon_days must be at least 1"):
        default_embargo_for_horizon(0)


def test_nw_lags_for_locked_horizons() -> None:
    """Reuse nw_lags_for_horizon: h → h-1 including zero for h=1."""
    assert nw_lags_for_horizon(1) == 0
    assert nw_lags_for_horizon(5) == 4
    assert nw_lags_for_horizon(21) == 20