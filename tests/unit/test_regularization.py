"""Tests for regularized linear volatility models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.modeling.regularization import (
    ElasticNetModel,
    LassoModel,
    RidgeModel,
)


SIGNAL_COLUMN = "signal"
NOISE_COLUMN = "noise"
N_ROWS = 200
N_TRAIN = 150


def _linear_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Build a synthetic design with one strong signal column."""
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
    # Intercept keeps RV-like positivity without destroying linearity.
    target = pd.Series(
        2.0 + 0.4 * signal + rng.normal(0.0, 0.01, N_ROWS),
        index=index,
        name="target_rv_cc_5d",
    )
    return features, target


def test_ridge_predict_shape_and_floor() -> None:
    """Ridge should return aligned positive predictions."""
    features, target = _linear_frame()
    model = RidgeModel().fit(features.iloc[:N_TRAIN], target.iloc[:N_TRAIN])
    predictions = model.predict(features.iloc[N_TRAIN:])
    assert predictions.shape[0] == N_ROWS - N_TRAIN
    assert list(predictions.index) == list(features.iloc[N_TRAIN:].index)
    assert (predictions >= 1e-8).all()


def test_ridge_recovers_signal_direction() -> None:
    """Ridge coefficient on the true signal column should dominate noise."""
    features, target = _linear_frame()
    model = RidgeModel(alpha=1.0).fit(features, target)
    coef = model.coefficients()
    assert abs(float(coef[SIGNAL_COLUMN])) > abs(float(coef[NOISE_COLUMN]))


def test_lasso_can_zero_noise_coefficient() -> None:
    """With enough penalty, Lasso should shrink pure noise toward zero."""
    features, target = _linear_frame()
    model = LassoModel(alpha=0.05).fit(features, target)
    coef = model.coefficients()
    assert abs(float(coef[SIGNAL_COLUMN])) > 0.0
    assert abs(float(coef[NOISE_COLUMN])) < 1e-6


def test_elasticnet_fit_predict() -> None:
    """ElasticNet should fit and predict without errors."""
    features, target = _linear_frame()
    model = ElasticNetModel().fit(features.iloc[:N_TRAIN], target.iloc[:N_TRAIN])
    predictions = model.predict(features.iloc[N_TRAIN:])
    assert predictions.shape[0] == N_ROWS - N_TRAIN
    assert (predictions > 0).all()


def test_predict_before_fit_raises() -> None:
    """Unfitted regularized models should raise on predict."""
    features, _ = _linear_frame()
    with pytest.raises(DataValidationError, match="must be fitted"):
        RidgeModel().predict(features)


def test_missing_columns_on_predict_raise() -> None:
    """Predict should require the same columns used at fit time."""
    features, target = _linear_frame()
    model = RidgeModel().fit(features, target)
    broken = features.drop(columns=[SIGNAL_COLUMN])
    with pytest.raises(DataValidationError, match="Missing feature columns"):
        model.predict(broken)


def test_feature_names_match_training_columns() -> None:
    """Fitted models should expose training feature names."""
    features, target = _linear_frame()
    model = RidgeModel().fit(features, target)
    assert model.feature_names() == tuple(features.columns)