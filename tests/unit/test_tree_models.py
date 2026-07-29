"""Tests for tree-based volatility models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.modeling.tree_models import RandomForestVolModel


SIGNAL_COLUMN = "signal"
NOISE_COLUMN = "noise"
N_ROWS = 300
N_TRAIN = 220


def _nonlinear_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Build a synthetic design with a nonlinear signal column."""
    index = pd.bdate_range("2024-01-02", periods=N_ROWS)
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 1.0, N_ROWS)
    noise = rng.normal(0.0, 1.0, N_ROWS)
    features = pd.DataFrame(
        {
            SIGNAL_COLUMN: signal,
            NOISE_COLUMN: noise,
            "other": rng.normal(0.0, 1.0, N_ROWS),
        },
        index=index,
    )
    # Threshold + square: linear models struggle; trees should prefer signal.
    nonlinear = np.where(signal > 0.0, signal**2, 0.1)
    target = pd.Series(
        0.05 + 0.4 * nonlinear + rng.normal(0.0, 0.01, N_ROWS),
        index=index,
        name="target_rv_cc_5d",
    )
    return features, target


def test_random_forest_predict_shape_and_floor() -> None:
    """Random forest should return aligned positive predictions."""
    features, target = _nonlinear_frame()
    model = RandomForestVolModel().fit(
        features.iloc[:N_TRAIN],
        target.iloc[:N_TRAIN],
    )
    predictions = model.predict(features.iloc[N_TRAIN:])
    assert predictions.shape[0] == N_ROWS - N_TRAIN
    assert list(predictions.index) == list(features.iloc[N_TRAIN:].index)
    assert (predictions >= 1e-8).all()


def test_random_forest_ranks_signal_above_noise() -> None:
    """Impurity importance of the true signal should dominate noise."""
    features, target = _nonlinear_frame()
    model = RandomForestVolModel().fit(features, target)
    importances = model.feature_importances()
    assert float(importances[SIGNAL_COLUMN]) > float(importances[NOISE_COLUMN])


def test_predict_before_fit_raises() -> None:
    """Unfitted tree models should raise on predict."""
    features, _ = _nonlinear_frame()
    with pytest.raises(DataValidationError, match="must be fitted"):
        RandomForestVolModel().predict(features)


def test_missing_columns_on_predict_raise() -> None:
    """Predict should require the same columns used at fit time."""
    features, target = _nonlinear_frame()
    model = RandomForestVolModel().fit(features, target)
    broken = features.drop(columns=[SIGNAL_COLUMN])
    with pytest.raises(DataValidationError, match="Missing feature columns"):
        model.predict(broken)


def test_feature_names_match_training_columns() -> None:
    """Fitted models should expose training feature names."""
    features, target = _nonlinear_frame()
    model = RandomForestVolModel().fit(features, target)
    assert model.feature_names() == tuple(features.columns)


def test_invalid_n_estimators_raises() -> None:
    """Constructor should reject non-positive n_estimators."""
    with pytest.raises(DataValidationError, match="n_estimators"):
        RandomForestVolModel(n_estimators=0)