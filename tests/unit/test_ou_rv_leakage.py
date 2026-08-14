"""Leakage and integrity tests for OuRvModel frozen-origin fit/predict.

Asserts train-only estimation, predict ignoring target, constant test-block
forecasts, ignored feature values, and no mutation of caller frames.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from vip.modeling.baselines import DEFAULT_OU_MIN_OBS, OuRvModel

N_ROWS = 60
TRAIN_END = 40
DUMMY_FEATURE_COLUMN = "rv_cc_1d"
FIRST_OBSERVATION = 0
SHUFFLE_SEED = 1
FEATURE_SHOCK = 100.0
TARGET_COLUMN_NAME = "target_rv_cc_5d"


def _positive_panel(n_rows: int = N_ROWS) -> tuple[pd.DataFrame, pd.Series]:
    """Build a strictly positive dummy feature/target panel.

    Parameters
    ----------
    n_rows : int, default 60
        Number of business days.

    Returns
    -------
    features : pandas.DataFrame
        Dummy feature column (ignored by OU except for the index).
    target : pandas.Series
        Strictly positive training target.
    """
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    features = pd.DataFrame(
        {DUMMY_FEATURE_COLUMN: np.linspace(0.01, 0.03, n_rows)},
        index=index,
    )
    target = pd.Series(
        np.linspace(0.02, 0.04, n_rows),
        index=index,
        name=TARGET_COLUMN_NAME,
    )
    return features, target


def test_fit_ignores_post_train_target() -> None:
    """Permuting target values after the train index must not change state."""
    features, target = _positive_panel()
    train_features = features.iloc[:TRAIN_END]
    train_target = target.iloc[:TRAIN_END]

    shocked = target.copy()
    future_values = shocked.iloc[TRAIN_END:].to_numpy(dtype=float, copy=True)
    rng = np.random.default_rng(SHUFFLE_SEED)
    rng.shuffle(future_values)
    shocked.iloc[TRAIN_END:] = future_values

    model_train_only = OuRvModel().fit(train_features, train_target)
    model_with_future = OuRvModel().fit(train_features, shocked)

    theta_a, phi_a, x_t_a = model_train_only.fitted_state()
    theta_b, phi_b, x_t_b = model_with_future.fitted_state()
    assert theta_a == pytest.approx(theta_b)
    assert phi_a == pytest.approx(phi_b)
    assert x_t_a == pytest.approx(x_t_b)

    test_features = features.iloc[TRAIN_END:]
    pd.testing.assert_series_equal(
        model_train_only.predict(test_features),
        model_with_future.predict(test_features),
    )


def test_predict_does_not_accept_or_use_target() -> None:
    """predict must not take target and must ignore a target-like column."""
    parameters = inspect.signature(OuRvModel.predict).parameters
    assert "target" not in parameters
    assert "features" in parameters

    features, target = _positive_panel()
    train_features = features.iloc[:TRAIN_END]
    test_features = features.iloc[TRAIN_END:]
    model = OuRvModel().fit(train_features, target.iloc[:TRAIN_END])

    dirty = test_features.copy()
    dirty[TARGET_COLUMN_NAME] = np.linspace(
        FEATURE_SHOCK,
        FEATURE_SHOCK * 2.0,
        len(dirty),
    )
    pd.testing.assert_series_equal(
        model.predict(test_features),
        model.predict(dirty),
    )


def test_frozen_origin_predictions_are_constant() -> None:
    """MVP frozen-origin forecasts must be constant across the test index."""
    features, target = _positive_panel()
    train_features = features.iloc[:TRAIN_END]
    test_features = features.iloc[TRAIN_END:]
    predictions = (
        OuRvModel()
        .fit(train_features, target.iloc[:TRAIN_END])
        .predict(test_features)
    )
    assert predictions.nunique() == 1
    pd.testing.assert_index_equal(predictions.index, test_features.index)
    assert float(predictions.to_numpy()[FIRST_OBSERVATION]) > 0.0


def test_fit_ignores_feature_column_values() -> None:
    """Scrambling feature values on the train index must not change state."""
    features, target = _positive_panel()
    train_features = features.iloc[:TRAIN_END]
    train_target = target.iloc[:TRAIN_END]
    shocked_features = train_features.copy()
    shocked_features[DUMMY_FEATURE_COLUMN] = FEATURE_SHOCK

    state_clean = OuRvModel().fit(train_features, train_target).fitted_state()
    state_shocked = OuRvModel().fit(shocked_features, train_target).fitted_state()
    assert state_clean[0] == pytest.approx(state_shocked[0])
    assert state_clean[1] == pytest.approx(state_shocked[1])
    assert state_clean[2] == pytest.approx(state_shocked[2])


def test_fit_does_not_mutate_caller_frames() -> None:
    """fit must not mutate the caller's features or target."""
    features, target = _positive_panel()
    features_before = features.copy()
    target_before = target.copy()
    OuRvModel().fit(features.iloc[:TRAIN_END], target)
    pd.testing.assert_frame_equal(features, features_before)
    pd.testing.assert_series_equal(target, target_before)


def test_default_min_obs_allows_flagship_length_train() -> None:
    """Default min_obs must accept a train window of that length."""
    features, target = _positive_panel(n_rows=DEFAULT_OU_MIN_OBS)
    model = OuRvModel().fit(features, target)
    predictions = model.predict(features)
    assert predictions.shape[0] == DEFAULT_OU_MIN_OBS
    assert predictions.nunique() == 1