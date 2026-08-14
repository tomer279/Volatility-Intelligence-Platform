"""Tests for OuRvModel discrete OU / AR(1) log-RV baseline.

Covers noiseless parameter recovery, exact h-step prediction, noisy AR(1)
phi recovery, prediction floor, and typed validation errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.modeling.baselines import (
    DEFAULT_OU_HORIZON_DAYS,
    DEFAULT_OU_MIN_OBS,
    DEFAULT_PREDICTION_FLOOR,
    LAG_SHIFT,
    LAST_OBSERVATION,
    MIN_OU_OBSERVATIONS,
    OuRvModel,
)

FIRST_OBSERVATION = 0
NOISE_SEED = 0
TRUE_THETA = -2.5
TRUE_PHI = 0.75
FIRST_LOG_DEVIATION = 1.25
N_NOISELESS = 48
N_NOISY = 400
NOISE_SCALE = 0.02
PHI_ABS_TOL = 0.05
THETA_ABS_TOL = 0.15
SHORT_N_OBS = 8
FLOOR_TEST_FLOOR = 0.5
DUMMY_FEATURE_COLUMN = "rv_cc_1d"


def _ar1_log_vol_series(
    n_obs: int,
    theta: float,
    phi: float,
    first_deviation: float,
    noise_scale: float = 0.0,
) -> pd.Series:
    """Build a positive RV series whose log follows a discrete AR(1).

    Parameters
    ----------
    n_obs : int
        Number of observations.
    theta : float
        Long-run mean of log-vol.
    phi : float
        AR(1) coefficient.
    first_deviation : float
        ``x_0 - theta``.
    noise_scale : float, default 0.0
        Gaussian innovation standard deviation on the log scale.

    Returns
    -------
    pandas.Series
        ``exp(x_t)`` on a business-day index.
    """
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.normal(0.0, noise_scale, n_obs)
    noise[FIRST_OBSERVATION] = 0.0
    log_values = np.empty(n_obs, dtype=float)
    log_values[FIRST_OBSERVATION] = theta + first_deviation
    for step in range(LAG_SHIFT, n_obs):
        previous = step - LAG_SHIFT
        log_values[step] = (
            theta
            + phi * (log_values[previous] - theta)
            + noise[step]
        )
    index = pd.bdate_range("2020-01-02", periods=n_obs)
    return pd.Series(np.exp(log_values), index=index, name="target_rv_cc_5d")


def _dummy_features(index: pd.Index) -> pd.DataFrame:
    """Return a dummy feature frame used only for index alignment."""
    ones = np.ones(len(index), dtype=float)
    return pd.DataFrame({DUMMY_FEATURE_COLUMN: ones}, index=index)


def test_noiseless_ar1_recovers_theta_phi_and_h_step() -> None:
    """Noiseless AR(1) on log-vol should recover (θ, φ, x_T) and the h-step."""
    target = _ar1_log_vol_series(
        N_NOISELESS,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
    )
    features = _dummy_features(target.index)
    model = OuRvModel(horizon_days=DEFAULT_OU_HORIZON_DAYS).fit(features, target)
    theta, phi, end_log_state = model.fitted_state()
    x_t = float(np.log(target.to_numpy(dtype=float)[LAST_OBSERVATION]))

    assert theta == pytest.approx(TRUE_THETA, abs=1e-8)
    assert phi == pytest.approx(TRUE_PHI, abs=1e-8)
    assert end_log_state == pytest.approx(x_t)

    expected_log = TRUE_THETA + (TRUE_PHI ** DEFAULT_OU_HORIZON_DAYS) * (
        x_t - TRUE_THETA
    )
    expected_y = max(float(np.exp(expected_log)), DEFAULT_PREDICTION_FLOOR)
    predictions = model.predict(features)
    assert predictions.nunique() == 1
    assert float(predictions.to_numpy()[FIRST_OBSERVATION]) == pytest.approx(
        expected_y
    )
    pd.testing.assert_index_equal(predictions.index, features.index)


def test_noisy_ar1_phi_near_truth() -> None:
    """Fitted φ on a long noisy AR(1) should be close to the DGP value."""
    target = _ar1_log_vol_series(
        N_NOISY,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
        noise_scale=NOISE_SCALE,
    )
    features = _dummy_features(target.index)
    model = OuRvModel().fit(features, target)
    theta, phi, _end_log_state = model.fitted_state()
    assert phi == pytest.approx(TRUE_PHI, abs=PHI_ABS_TOL)
    assert theta == pytest.approx(TRUE_THETA, abs=THETA_ABS_TOL)


def test_known_state_short_path_matches_h_step_formula() -> None:
    """Short noiseless path keeps x_T away from θ so h-step math is visible."""
    horizon_days = DEFAULT_OU_HORIZON_DAYS
    target = _ar1_log_vol_series(
        SHORT_N_OBS,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
    )
    features = _dummy_features(target.index)
    model = OuRvModel(
        horizon_days=horizon_days,
        min_obs=MIN_OU_OBSERVATIONS,
    ).fit(features, target)

    x_t = float(np.log(target.to_numpy(dtype=float)[LAST_OBSERVATION]))
    expected_log = TRUE_THETA + (TRUE_PHI ** horizon_days) * (x_t - TRUE_THETA)
    expected_y = max(float(np.exp(expected_log)), DEFAULT_PREDICTION_FLOOR)
    predicted = float(model.predict(features).to_numpy()[FIRST_OBSERVATION])
    assert predicted == pytest.approx(expected_y, rel=1e-8)


def test_horizon_days_changes_forecast() -> None:
    """h=1 and h=5 must differ when x_T is away from θ (not hard-coded 5)."""
    target = _ar1_log_vol_series(
        SHORT_N_OBS,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
    )
    features = _dummy_features(target.index)
    pred_h1 = (
        OuRvModel(horizon_days=1, min_obs=MIN_OU_OBSERVATIONS)
        .fit(features, target)
        .predict(features)
    )
    pred_h5 = (
        OuRvModel(horizon_days=DEFAULT_OU_HORIZON_DAYS, min_obs=MIN_OU_OBSERVATIONS)
        .fit(features, target)
        .predict(features)
    )
    assert float(pred_h1.to_numpy()[FIRST_OBSERVATION]) != pytest.approx(
        float(pred_h5.to_numpy()[FIRST_OBSERVATION])
    )


def test_predictions_respect_floor() -> None:
    """Tiny RV paths must be clipped to the configured prediction floor."""
    target = _ar1_log_vol_series(
        N_NOISELESS,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
    )
    features = _dummy_features(target.index)
    model = OuRvModel(prediction_floor=FLOOR_TEST_FLOOR).fit(features, target)
    predictions = model.predict(features)
    assert (predictions >= FLOOR_TEST_FLOOR).all()
    assert float(predictions.to_numpy()[FIRST_OBSERVATION]) == pytest.approx(
        FLOOR_TEST_FLOOR
    )


def test_empty_training_raises() -> None:
    """All-non-finite training targets must raise DataValidationError."""
    target = _ar1_log_vol_series(
        N_NOISELESS,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
    )
    features = _dummy_features(target.index)
    nan_target = pd.Series(np.full(N_NOISELESS, np.nan), index=target.index)
    with pytest.raises(DataValidationError, match="finite"):
        OuRvModel().fit(features, nan_target)


def test_non_positive_training_raises() -> None:
    """All-zero or all-negative targets must raise DataValidationError."""
    target = _ar1_log_vol_series(
        N_NOISELESS,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
    )
    features = _dummy_features(target.index)
    zeros = pd.Series(np.zeros(N_NOISELESS, dtype=float), index=target.index)
    negatives = -target
    with pytest.raises(DataValidationError, match="positive"):
        OuRvModel().fit(features, zeros)
    with pytest.raises(DataValidationError, match="positive"):
        OuRvModel().fit(features, negatives)


def test_zero_rows_are_dropped_when_enough_positive_remain() -> None:
    """Occasional zero RV (h=1 flat close) must not abort fit."""
    target = _ar1_log_vol_series(
        N_NOISELESS,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
    )
    mixed = target.copy()
    mixed.iloc[LAST_OBSERVATION] = 0.0
    features = _dummy_features(mixed.index)
    model = OuRvModel().fit(features, mixed)
    theta, phi, end_state = model.fitted_state()
    assert np.isfinite(theta)
    assert np.isfinite(phi)


def test_too_few_observations_raises() -> None:
    """Fewer than min_obs finite positive rows must raise."""
    n_short = DEFAULT_OU_MIN_OBS - 1
    target = _ar1_log_vol_series(
        n_short,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
    )
    features = _dummy_features(target.index)
    with pytest.raises(DataValidationError, match="at least"):
        OuRvModel().fit(features, target)


def test_unfitted_predict_raises() -> None:
    """Predict before fit must raise DataValidationError."""
    target = _ar1_log_vol_series(
        N_NOISELESS,
        TRUE_THETA,
        TRUE_PHI,
        FIRST_LOG_DEVIATION,
    )
    features = _dummy_features(target.index)
    with pytest.raises(DataValidationError, match="must be fitted"):
        OuRvModel().predict(features)


def test_invalid_constructor_args_raise() -> None:
    """Constructor must reject non-positive floor, h < 1, and tiny min_obs."""
    with pytest.raises(DataValidationError, match="horizon_days"):
        OuRvModel(horizon_days=0)
    with pytest.raises(DataValidationError, match="prediction_floor"):
        OuRvModel(prediction_floor=0.0)
    with pytest.raises(DataValidationError, match="min_obs"):
        OuRvModel(min_obs=MIN_OU_OBSERVATIONS - 1)