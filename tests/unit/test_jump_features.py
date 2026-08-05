"""Tests for daily jump-robust realized features and leakage alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.features.jump_features import build_jump_features
from vip.features.realized import (
    BIPOWER_SCALE,
    bipower_variation_trailing,
    bipower_volatility_trailing,
    jump_proportion_trailing,
    realized_variance_trailing,
)
from vip.features.registry import create_default_registry
from vip.features.targets import daily_log_returns

WINDOW_5D = 5
CUTOFF_POSITION = 40
SHOCK_OFFSET = 2
PAIR_COUNT_OFFSET = 1


def _synthetic_ohlcv(n_rows: int = 80) -> pd.DataFrame:
    """Build synthetic canonical OHLCV with a non-trivial return path."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    # Piecewise closes so |r_i||r_{i-1}| is not degenerate.
    noise = np.array([((-1.0) ** i) * (0.01 + 0.001 * i) for i in range(n_rows)])
    close = 100.0 * np.exp(np.cumsum(noise))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000.0, 3_000.0, n_rows),
        },
        index=index,
    )


def test_registry_lists_jump_when_opted_in() -> None:
    """Opt-in registry should expose the ``jump`` family name."""
    core = create_default_registry()
    assert "jump" not in core.list_names()

    with_jump = create_default_registry(include_jump=True)
    assert with_jump.list_names()[-1] == "jump"


def test_build_jump_features_columns() -> None:
    """Jump builder should emit jump-proportion columns only."""
    features = build_jump_features(_synthetic_ohlcv())
    assert list(features.columns) == [
        "jump_prop_1d",
        "jump_prop_5d",
        "jump_prop_21d",
    ]


def test_bipower_5d_matches_manual_pairs() -> None:
    """5d bipower variation should equal (π/2) times four adjacent products."""
    returns = daily_log_returns(_synthetic_ohlcv()["close"])
    end_idx = 20
    window = WINDOW_5D
    pair_count = window - PAIR_COUNT_OFFSET

    abs_ret = returns.abs()
    products = abs_ret * abs_ret.shift(1)
    start_idx = end_idx - pair_count + 1
    expected = float(BIPOWER_SCALE * products.iloc[start_idx : end_idx + 1].sum())

    actual = bipower_variation_trailing(returns, window)
    assert actual.iloc[end_idx] == pytest.approx(expected)


def test_jump_proportion_zero_guards_and_clip() -> None:
    """Jump proportion clamps negative RV-BPV gaps and guards zero RV."""
    index = pd.bdate_range("2024-01-02", periods=6)
    # Construct returns where BPV can exceed RV in finite samples.
    returns = pd.Series(
        [np.nan, 0.0, 0.02, -0.02, 0.01, -0.01],
        index=index,
        name="r",
    )
    window = 2
    rv = realized_variance_trailing(returns, window)
    bpv = bipower_variation_trailing(returns, window)
    prop = jump_proportion_trailing(returns, window)

    valid = rv.notna()
    assert (prop.loc[valid] >= 0.0).all()
    assert (prop.loc[valid] <= 1.0).all()
    # Where RV == 0, proportion must be 0.
    zero_mask = valid & (rv == 0.0)
    if zero_mask.any():
        assert (prop.loc[zero_mask] == 0.0).all()
    # Where RV > 0, equals clipped gap / RV.
    positive = valid & (rv > 0.0)
    expected = ((rv - bpv).clip(lower=0.0) / rv).loc[positive]
    pd.testing.assert_series_equal(
        prop.loc[positive],
        expected,
        check_names=False,
    )


def test_jump_features_unchanged_when_future_truncated() -> None:
    """Features at t must equal features built from data through t only."""
    full = _synthetic_ohlcv()
    registry = create_default_registry(include_jump=True)
    full_features = registry.build_all(full, names=["jump"])

    cutoff_date = full.index[CUTOFF_POSITION]
    truncated = full.iloc[: CUTOFF_POSITION + 1]
    truncated_features = registry.build_all(truncated, names=["jump"])

    pd.testing.assert_series_equal(
        full_features.loc[cutoff_date],
        truncated_features.loc[cutoff_date],
        check_names=False,
    )


def test_future_close_shock_does_not_change_jump_features_at_t() -> None:
    """A shock after t must not change jump features at t."""
    base = _synthetic_ohlcv()
    registry = create_default_registry(include_jump=True)

    t_date = base.index[CUTOFF_POSITION]
    shock_date = base.index[CUTOFF_POSITION + SHOCK_OFFSET]
    shocked = base.copy()
    shocked.loc[shock_date, "close"] = shocked.loc[shock_date, "close"] * 1.25

    base_features = registry.build_all(base, names=["jump"])
    shocked_features = registry.build_all(shocked, names=["jump"])
    pd.testing.assert_series_equal(
        base_features.loc[t_date],
        shocked_features.loc[t_date],
        check_names=False,
    )


def test_helper_values_match_prefix_slice_at_t() -> None:
    """Helper at t must match recomputation on returns[:t] (inclusive)."""
    returns = daily_log_returns(_synthetic_ohlcv()["close"])
    t_idx = CUTOFF_POSITION
    window = WINDOW_5D

    full_bpv = bipower_volatility_trailing(returns, window)
    full_jump = jump_proportion_trailing(returns, window)
    prefix = returns.iloc[: t_idx + 1]
    prefix_bpv = bipower_volatility_trailing(prefix, window)
    prefix_jump = jump_proportion_trailing(prefix, window)

    assert full_bpv.iloc[t_idx] == pytest.approx(prefix_bpv.iloc[t_idx])
    assert full_jump.iloc[t_idx] == pytest.approx(prefix_jump.iloc[t_idx])


def test_invalid_window_raises() -> None:
    """Windows below 1 should raise DataValidationError."""
    returns = daily_log_returns(_synthetic_ohlcv()["close"])
    with pytest.raises(DataValidationError, match="Window must be at least 1"):
        bipower_variation_trailing(returns, 0)


def test_jump_family_excludes_bpv_level_columns() -> None:
    """Screening jump family must not duplicate HAR rv_cc_* levels."""
    features = build_jump_features(_synthetic_ohlcv())
    assert not any(name.startswith("bpv_cc_") for name in features.columns)