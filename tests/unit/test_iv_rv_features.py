"""Unit and leakage tests for IV−RV gap feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.features.iv_rv_features import (
    ANNUALIZATION_DAYS,
    PERCENT_TO_FRACTION,
    VIX_MINUS_RV_1D_COLUMN,
    VIX_MINUS_RV_5D_COLUMN,
    VIX_MINUS_RV_21D_COLUMN,
    VIX_RV_RATIO_5D_COLUMN,
    VIX_VOL_DAILY_COLUMN,
    build_iv_rv_features,
    vix_level_to_daily_vol,
)

CUTOFF_POSITION = 10
SHOCK_OFFSET = 2
N_ROWS = 30
VIX_LEVEL_KNOWN = 20.0


def _synthetic_har_and_vix(
    n_rows: int = N_ROWS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build aligned synthetic HAR RV columns and VIX levels."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    har_frame = pd.DataFrame(
        {
            "rv_cc_1d": np.linspace(0.005, 0.020, n_rows),
            "rv_cc_5d": np.linspace(0.010, 0.040, n_rows),
            "rv_cc_21d": np.linspace(0.020, 0.060, n_rows),
        },
        index=index,
    )
    vix_level = pd.Series(
        np.linspace(12.0, 28.0, n_rows),
        index=index,
        name="vix_level",
    )
    return har_frame, vix_level


def test_vix_level_to_daily_vol_known_value() -> None:
    """Locked conversion: 20 → (20/100)/sqrt(252)."""
    index = pd.bdate_range("2024-01-02", periods=3)
    vix_level = pd.Series([VIX_LEVEL_KNOWN] * 3, index=index)
    expected = (VIX_LEVEL_KNOWN / PERCENT_TO_FRACTION) / np.sqrt(
        ANNUALIZATION_DAYS
    )
    actual = vix_level_to_daily_vol(vix_level)
    assert actual.name == VIX_VOL_DAILY_COLUMN
    assert actual.iloc[0] == pytest.approx(expected)
    assert (actual == expected).all()


def test_build_iv_rv_features_gap_arithmetic() -> None:
    """Gaps and ratio match pointwise daily-vol minus trailing RV."""
    har_frame, vix_level = _synthetic_har_and_vix()
    features = build_iv_rv_features(har_frame, vix_level)
    vix_vol = vix_level_to_daily_vol(vix_level)

    assert list(features.columns) == [
        VIX_VOL_DAILY_COLUMN,
        VIX_MINUS_RV_1D_COLUMN,
        VIX_MINUS_RV_5D_COLUMN,
        VIX_MINUS_RV_21D_COLUMN,
        VIX_RV_RATIO_5D_COLUMN,
    ]
    pd.testing.assert_series_equal(
        features[VIX_VOL_DAILY_COLUMN],
        vix_vol,
        check_names=True,
    )
    pd.testing.assert_series_equal(
        features[VIX_MINUS_RV_1D_COLUMN],
        vix_vol - har_frame["rv_cc_1d"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        features[VIX_MINUS_RV_5D_COLUMN],
        vix_vol - har_frame["rv_cc_5d"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        features[VIX_MINUS_RV_21D_COLUMN],
        vix_vol - har_frame["rv_cc_21d"],
        check_names=False,
    )
    expected_ratio = vix_vol / har_frame["rv_cc_5d"]
    pd.testing.assert_series_equal(
        features[VIX_RV_RATIO_5D_COLUMN],
        expected_ratio,
        check_names=False,
    )


def test_vix_rv_ratio_5d_zero_guard() -> None:
    """Ratio is NaN when trailing rv_cc_5d is zero."""
    har_frame, vix_level = _synthetic_har_and_vix(n_rows=5)
    har_frame = har_frame.copy()
    zero_date = har_frame.index[2]
    har_frame.loc[zero_date, "rv_cc_5d"] = 0.0
    features = build_iv_rv_features(har_frame, vix_level)
    assert np.isnan(features.loc[zero_date, VIX_RV_RATIO_5D_COLUMN])


def test_missing_har_columns_raise() -> None:
    """Missing rv_cc_* columns must raise DataValidationError."""
    index = pd.bdate_range("2024-01-02", periods=5)
    har_frame = pd.DataFrame({"rv_cc_1d": np.ones(5)}, index=index)
    vix_level = pd.Series(np.full(5, VIX_LEVEL_KNOWN), index=index)
    with pytest.raises(DataValidationError, match="missing required columns"):
        build_iv_rv_features(har_frame, vix_level)


def test_future_vix_permutation_does_not_change_features_at_t() -> None:
    """Permuting vix_level after t must not change IV−RV features at t."""
    har_frame, vix_level = _synthetic_har_and_vix()
    t_date = har_frame.index[CUTOFF_POSITION]
    shock_date = har_frame.index[CUTOFF_POSITION + SHOCK_OFFSET]

    shocked_vix = vix_level.copy()
    shocked_vix.loc[shock_date] = shocked_vix.loc[shock_date] * 2.0

    base_feat = build_iv_rv_features(har_frame, vix_level)
    shocked_feat = build_iv_rv_features(har_frame, shocked_vix)
    pd.testing.assert_series_equal(
        base_feat.loc[t_date],
        shocked_feat.loc[t_date],
        check_names=False,
    )


def test_future_rv_shock_does_not_change_features_at_t() -> None:
    """Shocking trailing RV after t must not change gaps at t."""
    har_frame, vix_level = _synthetic_har_and_vix()
    t_date = har_frame.index[CUTOFF_POSITION]
    shock_date = har_frame.index[CUTOFF_POSITION + SHOCK_OFFSET]

    shocked_har = har_frame.copy()
    shocked_har.loc[shock_date, "rv_cc_5d"] = (
        shocked_har.loc[shock_date, "rv_cc_5d"] * 3.0
    )

    base_feat = build_iv_rv_features(har_frame, vix_level)
    shocked_feat = build_iv_rv_features(shocked_har, vix_level)
    pd.testing.assert_series_equal(
        base_feat.loc[t_date],
        shocked_feat.loc[t_date],
        check_names=False,
    )


def test_gap_at_t_equals_pointwise_no_forward_shift() -> None:
    """Gap at t uses vix_vol_daily[t] and rv_cc_*[t] only (no shift)."""
    har_frame, vix_level = _synthetic_har_and_vix()
    features = build_iv_rv_features(har_frame, vix_level)
    t_date = har_frame.index[CUTOFF_POSITION]
    vix_vol_t = float(vix_level_to_daily_vol(vix_level).loc[t_date])

    assert features.loc[t_date, VIX_MINUS_RV_1D_COLUMN] == pytest.approx(
        vix_vol_t - float(har_frame.loc[t_date, "rv_cc_1d"])
    )
    assert features.loc[t_date, VIX_MINUS_RV_5D_COLUMN] == pytest.approx(
        vix_vol_t - float(har_frame.loc[t_date, "rv_cc_5d"])
    )
    assert features.loc[t_date, VIX_MINUS_RV_21D_COLUMN] == pytest.approx(
        vix_vol_t - float(har_frame.loc[t_date, "rv_cc_21d"])
    )
    # Explicitly not using a lead of RV.
    next_date = har_frame.index[CUTOFF_POSITION + 1]
    assert features.loc[t_date, VIX_MINUS_RV_5D_COLUMN] != pytest.approx(
        vix_vol_t - float(har_frame.loc[next_date, "rv_cc_5d"])
    )


def test_target_columns_are_ignored() -> None:
    """Presence of target_rv_cc_* must not change builder output."""
    har_frame, vix_level = _synthetic_har_and_vix()
    with_target = har_frame.copy()
    with_target["target_rv_cc_5d"] = np.linspace(0.03, 0.08, len(har_frame))

    base_feat = build_iv_rv_features(har_frame, vix_level)
    with_target_feat = build_iv_rv_features(with_target, vix_level)
    pd.testing.assert_frame_equal(base_feat, with_target_feat)


def test_builder_does_not_mutate_inputs() -> None:
    """Builder must leave caller har_frame and vix_level unchanged."""
    har_frame, vix_level = _synthetic_har_and_vix()
    har_before = har_frame.copy(deep=True)
    vix_before = vix_level.copy(deep=True)
    build_iv_rv_features(har_frame, vix_level)
    pd.testing.assert_frame_equal(har_frame, har_before)
    pd.testing.assert_series_equal(vix_level, vix_before)