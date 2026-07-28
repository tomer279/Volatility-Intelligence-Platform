"""Tests for baseline volatility models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.modeling.baselines import EwmaModel, HarRvOlsModel, HistoricalMeanModel


def _toy_frame(n_rows: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """Build a synthetic feature matrix and target."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "rv_cc_1d": rng.uniform(0.01, 0.05, n_rows),
            "rv_cc_5d": rng.uniform(0.02, 0.06, n_rows),
            "rv_cc_21d": rng.uniform(0.03, 0.07, n_rows),
            "ret_1d": rng.normal(0.0, 0.01, n_rows),
        },
        index=index,
    )
    target = (
        0.1
        + 0.4 * features["rv_cc_1d"]
        + 0.3 * features["rv_cc_5d"]
        + 0.2 * features["rv_cc_21d"]
    )
    return features, target


def test_historical_mean_predicts_training_mean() -> None:
    """Historical mean should forecast the training-target average."""
    features, target = _toy_frame()
    model = HistoricalMeanModel().fit(features.iloc[:40], target.iloc[:40])
    predictions = model.predict(features.iloc[40:])
    assert predictions.nunique() == 1
    assert predictions.iloc[0] == pytest.approx(float(target.iloc[:40].mean()))


def test_ewma_predicts_frozen_train_level() -> None:
    """EWMA predictions should be constant and equal the end-of-train level."""
    features, target = _toy_frame()
    model = EwmaModel(decay=0.9).fit(features.iloc[:40], target.iloc[:40])
    predictions = model.predict(features.iloc[40:])
    assert predictions.nunique() == 1
    assert predictions.iloc[0] > 0


def test_har_ols_recovers_near_true_coefficients() -> None:
    """HAR OLS should recover coefficients on a nearly noiseless design."""
    features, target = _toy_frame()
    model = HarRvOlsModel().fit(features, target)
    predictions = model.predict(features)
    assert predictions.shape[0] == features.shape[0]
    assert (predictions > 0).all()
    # In-sample fit should be very strong on this synthetic design.
    residuals = target - predictions
    assert float(np.mean(residuals**2)) < 1e-6


def test_har_missing_columns_raise() -> None:
    """HAR model should require the configured feature columns."""
    features, target = _toy_frame()
    broken = features.drop(columns=["rv_cc_21d"])
    with pytest.raises(DataValidationError, match="Missing HAR feature columns"):
        HarRvOlsModel().fit(broken, target)


def test_predict_before_fit_raises() -> None:
    """Unfitted models should raise on predict."""
    features, _ = _toy_frame()
    with pytest.raises(DataValidationError, match="must be fitted"):
        HistoricalMeanModel().predict(features)