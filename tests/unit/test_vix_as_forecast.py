"""Tests for VixAsForecastModel and registry name vix_as_forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from vip.domain.errors import DataValidationError
from vip.features.cross_asset import VIX_LEVEL_COLUMN
from vip.features.iv_rv_features import (
    VIX_VOL_DAILY_COLUMN,
    vix_level_to_daily_vol,
)
from vip.modeling.baselines import (
    DEFAULT_PREDICTION_FLOOR,
    VixAsForecastModel,
)
from vip.modeling.registry import create_default_model_registry

N_ROWS = 20
TRUE_INTERCEPT = 0.01
TRUE_SLOPE = 2.0
FLOOR_TEST_FLOOR = 0.5


def _panel_with_vix_vol_daily(
    n_rows: int = N_ROWS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a noiseless OLS panel on ``vix_vol_daily``."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    vix_vol = pd.Series(
        np.linspace(0.005, 0.025, n_rows),
        index=index,
        name=VIX_VOL_DAILY_COLUMN,
    )
    features = pd.DataFrame({VIX_VOL_DAILY_COLUMN: vix_vol}, index=index)
    target = TRUE_INTERCEPT + TRUE_SLOPE * vix_vol
    target = target.rename("target_rv_cc_5d")
    return features, target


def test_registry_includes_vix_as_forecast() -> None:
    """Default registry must expose ``vix_as_forecast``."""
    registry = create_default_model_registry()
    assert "vix_as_forecast" in registry.list_names()
    model = registry.create("vix_as_forecast")
    assert isinstance(model, VixAsForecastModel)


def test_ols_recovers_known_coefficients() -> None:
    """Intercept OLS should recover the synthetic intercept and slope."""
    features, target = _panel_with_vix_vol_daily()
    model = VixAsForecastModel().fit(features, target)
    predictions = model.predict(features)

    design = sm.add_constant(features[[VIX_VOL_DAILY_COLUMN]], has_constant="add")
    expected_params = sm.OLS(target, design).fit().params
    raw = design.to_numpy(dtype=float) @ expected_params.to_numpy(dtype=float)
    expected = pd.Series(
        np.maximum(raw, DEFAULT_PREDICTION_FLOOR),
        index=features.index,
        name="prediction",
    )
    pd.testing.assert_series_equal(predictions, expected)
    assert predictions.iloc[0] == pytest.approx(
        TRUE_INTERCEPT + TRUE_SLOPE * float(features[VIX_VOL_DAILY_COLUMN].iloc[0])
    )


def test_predict_derives_from_vix_level_when_vol_absent() -> None:
    """Predict should convert ``vix_level`` when ``vix_vol_daily`` is missing."""
    features_vol, target = _panel_with_vix_vol_daily()
    model = VixAsForecastModel().fit(features_vol, target)

    vix_level = features_vol[VIX_VOL_DAILY_COLUMN] * 100.0 * np.sqrt(252)
    features_level = pd.DataFrame(
        {VIX_LEVEL_COLUMN: vix_level},
        index=features_vol.index,
    )
    from_level = model.predict(features_level)
    from_vol = model.predict(features_vol)
    pd.testing.assert_series_equal(from_level, from_vol)

    derived = vix_level_to_daily_vol(features_level[VIX_LEVEL_COLUMN])
    pd.testing.assert_series_equal(
        derived,
        features_vol[VIX_VOL_DAILY_COLUMN],
        check_names=False,
    )


def test_missing_vix_columns_raise() -> None:
    """Fit/predict require ``vix_vol_daily`` or ``vix_level``."""
    index = pd.bdate_range("2024-01-02", periods=N_ROWS)
    features = pd.DataFrame({"rv_cc_1d": np.ones(N_ROWS)}, index=index)
    target = pd.Series(np.ones(N_ROWS), index=index)
    with pytest.raises(DataValidationError, match="vix_vol_daily|vix_level"):
        VixAsForecastModel().fit(features, target)

    fitted_features, fitted_target = _panel_with_vix_vol_daily()
    model = VixAsForecastModel().fit(fitted_features, fitted_target)
    with pytest.raises(DataValidationError, match="vix_vol_daily|vix_level"):
        model.predict(features)


def test_unfitted_predict_raises() -> None:
    """Predict before fit must raise DataValidationError."""
    features, _ = _panel_with_vix_vol_daily()
    with pytest.raises(DataValidationError, match="must be fitted"):
        VixAsForecastModel().predict(features)


def test_empty_training_raises() -> None:
    """All-NaN predictor/target rows must raise on fit."""
    index = pd.bdate_range("2024-01-02", periods=N_ROWS)
    features = pd.DataFrame(
        {VIX_VOL_DAILY_COLUMN: np.full(N_ROWS, np.nan)},
        index=index,
    )
    target = pd.Series(np.ones(N_ROWS), index=index)
    with pytest.raises(DataValidationError, match="No finite rows"):
        VixAsForecastModel().fit(features, target)


def test_predictions_respect_floor() -> None:
    """Negative raw OLS output must be clipped to the prediction floor."""
    index = pd.bdate_range("2024-01-02", periods=N_ROWS)
    vix_vol = pd.Series(
        np.linspace(0.001, 0.002, N_ROWS),
        index=index,
        name=VIX_VOL_DAILY_COLUMN,
    )
    features = pd.DataFrame({VIX_VOL_DAILY_COLUMN: vix_vol}, index=index)
    # Strongly negative slope so extrapolation below the floor is easy.
    target = 0.05 - 20.0 * vix_vol
    model = VixAsForecastModel(prediction_floor=FLOOR_TEST_FLOOR).fit(
        features,
        target,
    )
    high_vol = pd.DataFrame(
        {VIX_VOL_DAILY_COLUMN: np.full(N_ROWS, 1.0)},
        index=index,
    )
    predictions = model.predict(high_vol)
    assert (predictions >= FLOOR_TEST_FLOOR).all()
    assert predictions.iloc[0] == pytest.approx(FLOOR_TEST_FLOOR)


def test_fit_derives_from_vix_level_when_vol_absent() -> None:
    """Fit should convert ``vix_level`` when ``vix_vol_daily`` is missing."""
    features_vol, target = _panel_with_vix_vol_daily()
    vix_level = features_vol[VIX_VOL_DAILY_COLUMN] * 100.0 * np.sqrt(252)
    features_level = pd.DataFrame(
        {VIX_LEVEL_COLUMN: vix_level},
        index=features_vol.index,
    )

    model_from_level = VixAsForecastModel().fit(features_level, target)
    model_from_vol = VixAsForecastModel().fit(features_vol, target)

    preds_level = model_from_level.predict(features_vol)
    preds_vol = model_from_vol.predict(features_vol)
    pd.testing.assert_series_equal(preds_level, preds_vol)