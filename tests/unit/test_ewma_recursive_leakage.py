"""Leakage / integrity tests for EwmaRecursiveModel.

Asserts train-only fit, predict ignores target, observation-driven updates,
and no mutation of caller frames.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from vip.modeling.parametric import (
    DEFAULT_RECURSIVE_OBS_COLUMN,
    EwmaRecursiveModel,
)

N_ROWS = 60
TRAIN_END = 40
OBS_COLUMN = DEFAULT_RECURSIVE_OBS_COLUMN
TARGET_NAME = "target_rv_cc_5d"
SHUFFLE_SEED = 3
FEATURE_SHOCK = 100.0


def _panel(n_rows: int = N_ROWS) -> tuple[pd.DataFrame, pd.Series]:
    """Positive feature/target panel with trailing RV column."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    features = pd.DataFrame(
        {OBS_COLUMN: np.linspace(0.01, 0.03, n_rows)},
        index=index,
    )
    target = pd.Series(
        np.linspace(0.02, 0.04, n_rows),
        index=index,
        name=TARGET_NAME,
    )
    return features, target


def test_fit_ignores_post_train_target() -> None:
    """Permuting target after train index must not change fitted state."""
    features, target = _panel()
    train_features = features.iloc[:TRAIN_END]
    train_target = target.iloc[:TRAIN_END]

    shocked = target.copy()
    future = shocked.iloc[TRAIN_END:].to_numpy(dtype=float, copy=True)
    rng = np.random.default_rng(SHUFFLE_SEED)
    rng.shuffle(future)
    shocked.iloc[TRAIN_END:] = future

    model_a = EwmaRecursiveModel().fit(train_features, train_target)
    model_b = EwmaRecursiveModel().fit(train_features, shocked)
    assert model_a.fitted_decay() == pytest.approx(model_b.fitted_decay())
    assert model_a.fitted_level() == pytest.approx(model_b.fitted_level())


def test_predict_does_not_accept_or_use_target() -> None:
    """predict must not take target; label-like columns are ignored."""
    parameters = inspect.signature(EwmaRecursiveModel.predict).parameters
    assert "target" not in parameters
    assert "features" in parameters

    features, target = _panel()
    model = EwmaRecursiveModel().fit(
        features.iloc[:TRAIN_END],
        target.iloc[:TRAIN_END],
    )
    test_features = features.iloc[TRAIN_END:]
    dirty = test_features.copy()
    dirty[TARGET_NAME] = np.linspace(
        FEATURE_SHOCK,
        FEATURE_SHOCK * 2.0,
        len(dirty),
    )
    pd.testing.assert_series_equal(
        model.predict(test_features),
        model.predict(dirty),
    )


def test_observation_permutation_changes_predictions() -> None:
    """Shuffling trailing RV on the test block must change forecasts."""
    features, target = _panel()
    model = EwmaRecursiveModel().fit(
        features.iloc[:TRAIN_END],
        target.iloc[:TRAIN_END],
    )
    test_features = features.iloc[TRAIN_END:]
    shuffled = test_features.copy()
    values = shuffled[OBS_COLUMN].to_numpy(dtype=float, copy=True)
    rng = np.random.default_rng(SHUFFLE_SEED)
    rng.shuffle(values)
    shuffled[OBS_COLUMN] = values
    assert not np.allclose(
        model.predict(test_features).to_numpy(dtype=float),
        model.predict(shuffled).to_numpy(dtype=float),
    )


def test_fit_predict_do_not_mutate_inputs() -> None:
    """Caller frames/series must be unchanged after fit/predict."""
    features, target = _panel()
    train_features = features.iloc[:TRAIN_END].copy()
    train_target = target.iloc[:TRAIN_END].copy()
    features_before = train_features.copy()
    target_before = train_target.copy()
    model = EwmaRecursiveModel().fit(train_features, train_target)
    pd.testing.assert_frame_equal(train_features, features_before)
    pd.testing.assert_series_equal(train_target, target_before)

    test_features = features.iloc[TRAIN_END:].copy()
    test_before = test_features.copy()
    _ = model.predict(test_features)
    pd.testing.assert_frame_equal(test_features, test_before)