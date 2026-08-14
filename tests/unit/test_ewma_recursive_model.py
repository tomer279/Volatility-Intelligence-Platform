"""Tests for EwmaRecursiveModel train-fit decay and recursive predict.

Covers decay selection, recursive non-constant forecasts, floor, typed
errors, and distinctness from frozen EwmaModel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.modeling.baselines import EwmaModel
from vip.modeling.parametric import (
    DEFAULT_PREDICTION_FLOOR,
    DEFAULT_RECURSIVE_MIN_OBS,
    DEFAULT_RECURSIVE_OBS_COLUMN,
    EwmaRecursiveModel,
   )

N_TRAIN = 80
N_TEST = 12
OBS_COLUMN = DEFAULT_RECURSIVE_OBS_COLUMN
TARGET_NAME = "target_rv_cc_5d"
FLOOR_TEST_FLOOR = 0.5


def _train_panel(n_train: int = N_TRAIN) -> tuple[pd.DataFrame, pd.Series]:
    """Build a positive train panel with varying trailing RV."""
    index = pd.bdate_range("2020-01-02", periods=n_train)
    features = pd.DataFrame(
        {OBS_COLUMN: np.linspace(0.01, 0.04, n_train)},
        index=index,
    )
    target = pd.Series(
        np.linspace(0.015, 0.035, n_train),
        index=index,
        name=TARGET_NAME,
    )
    return features, target


def test_fit_selects_decay_and_end_level() -> None:
    """Fitted decay is in (0, 1) and end level is finite."""
    features, target = _train_panel()
    model = EwmaRecursiveModel().fit(features, target)
    decay = model.fitted_decay()
    level = model.fitted_level()
    assert 0.0 < decay < 1.0
    assert np.isfinite(level)
    assert level > 0.0


def test_predict_is_non_constant_when_obs_vary() -> None:
    """Recursive updates must move the forecast across the test block."""
    train_features, train_target = _train_panel()
    model = EwmaRecursiveModel().fit(train_features, train_target)
    test_index = pd.bdate_range("2020-06-01", periods=N_TEST)
    test_features = pd.DataFrame(
        {OBS_COLUMN: np.linspace(0.05, 0.01, N_TEST)},
        index=test_index,
    )
    preds = model.predict(test_features)
    assert len(preds.unique()) > 1


def test_distinct_from_frozen_ewma() -> None:
    """Same train data: recursive path differs from frozen constant ewma."""
    train_features, train_target = _train_panel()
    frozen = EwmaModel().fit(train_features, train_target)
    recursive = EwmaRecursiveModel().fit(train_features, train_target)
    test_index = pd.bdate_range("2020-06-01", periods=N_TEST)
    test_features = pd.DataFrame(
        {OBS_COLUMN: np.linspace(0.05, 0.01, N_TEST)},
        index=test_index,
    )
    frozen_preds = frozen.predict(test_features)
    recursive_preds = recursive.predict(test_features)
    assert not np.allclose(
        frozen_preds.to_numpy(dtype=float),
        recursive_preds.to_numpy(dtype=float),
    )


def test_prediction_floor_applied() -> None:
    """Predictions must respect prediction_floor."""
    train_features, train_target = _train_panel()
    tiny = train_target * 1e-12
    model = EwmaRecursiveModel(prediction_floor=FLOOR_TEST_FLOOR).fit(
        train_features,
        tiny,
    )
    test_features = pd.DataFrame(
        {OBS_COLUMN: np.full(N_TEST, 1e-12)},
        index=pd.bdate_range("2020-06-01", periods=N_TEST),
    )
    preds = model.predict(test_features)
    assert (preds >= FLOOR_TEST_FLOOR - 1e-15).all()


def test_unfitted_predict_raises() -> None:
    """predict before fit raises DataValidationError."""
    model = EwmaRecursiveModel()
    features = pd.DataFrame(
        {OBS_COLUMN: [0.01, 0.02]},
        index=pd.bdate_range("2020-01-02", periods=2),
    )
    with pytest.raises(DataValidationError, match="fitted"):
        model.predict(features)


def test_missing_observation_column_raises() -> None:
    """predict without observation_column raises."""
    train_features, train_target = _train_panel()
    model = EwmaRecursiveModel().fit(train_features, train_target)
    bad = pd.DataFrame(
        {"other": [0.01, 0.02]},
        index=pd.bdate_range("2020-06-01", periods=2),
    )
    with pytest.raises(DataValidationError, match="observation"):
        model.predict(bad)


def test_short_train_raises() -> None:
    """Fewer than min_obs finite targets raises."""
    n_short = DEFAULT_RECURSIVE_MIN_OBS - 1
    index = pd.bdate_range("2020-01-02", periods=n_short)
    features = pd.DataFrame({OBS_COLUMN: np.ones(n_short) * 0.02}, index=index)
    target = pd.Series(np.ones(n_short) * 0.02, index=index)
    with pytest.raises(DataValidationError, match="at least"):
        EwmaRecursiveModel().fit(features, target)


def test_default_floor_constant() -> None:
    """Locked prediction floor matches other baselines."""
    assert DEFAULT_PREDICTION_FLOOR == 1e-8