"""Tests for the model registry and walk-forward smoke checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.evaluation.comparison import summarize_walk_forward
from vip.evaluation.walk_forward import run_walk_forward
from vip.modeling.registry import create_default_model_registry

N_ROWS = 240
N_SPLITS = 4
EMBARGO_SIZE = 5


def _synthetic_design() -> tuple[pd.DataFrame, pd.Series]:
    """Build a synthetic design where linear RV features predict the target."""
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            "rv_cc_1d": rng.uniform(0.01, 0.05, N_ROWS),
            "rv_cc_5d": rng.uniform(0.02, 0.06, N_ROWS),
            "rv_cc_21d": rng.uniform(0.03, 0.07, N_ROWS),
            "noise": rng.normal(0.0, 1.0, N_ROWS),
        },
        index=index,
    )
    target = (
        0.05
        + 0.5 * features["rv_cc_1d"]
        + 0.3 * features["rv_cc_5d"]
        + 0.2 * features["rv_cc_21d"]
        + rng.normal(0.0, 0.001, N_ROWS)
    )
    return features, pd.Series(target, index=index, name="target_rv_cc_5d")


def _nonlinear_design() -> tuple[pd.DataFrame, pd.Series]:
    """Build a design where a thresholded signal drives the target."""
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    rng = np.random.default_rng(7)
    signal = rng.normal(0.0, 1.0, N_ROWS)
    features = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(0.0, 1.0, N_ROWS),
            "other": rng.normal(0.0, 1.0, N_ROWS),
        },
        index=index,
    )
    nonlinear = np.where(signal > 0.0, signal**2, 0.1)
    target = 0.05 + 0.4 * nonlinear + rng.normal(0.0, 0.01, N_ROWS)
    return features, pd.Series(target, index=index, name="target_rv_cc_5d")


def test_default_registry_contains_expected_models() -> None:
    """Default registry should expose baseline, linear, and tree model names."""
    registry = create_default_model_registry()
    names = registry.list_names()
    for expected in (
        "historical_mean",
        "ewma",
        "har_rv_ols",
        "ridge",
        "lasso",
        "elasticnet",
        "random_forest",
    ):
        assert expected in names


def test_unknown_model_raises() -> None:
    """Looking up an unknown model name should raise DataValidationError."""
    registry = create_default_model_registry()
    with pytest.raises(DataValidationError, match="Unknown model"):
        registry.get("not_a_real_model")


def test_create_many_returns_distinct_instances() -> None:
    """create_many should return fresh instances per call."""
    registry = create_default_model_registry()
    first = registry.create_many(["ridge", "historical_mean"])
    second = registry.create_many(["ridge", "historical_mean"])
    assert set(first) == {"ridge", "historical_mean"}
    assert first["ridge"] is not second["ridge"]


def test_ridge_beats_mean_on_walk_forward() -> None:
    """Ridge should beat historical mean on QLIKE in walk-forward."""
    features, target = _synthetic_design()
    registry = create_default_model_registry()
    models = registry.create_many(["historical_mean", "ridge"])

    fold_metrics = run_walk_forward(
        features=features,
        target=target,
        models=models,
        n_splits=N_SPLITS,
        embargo_size=EMBARGO_SIZE,
    )
    summary = summarize_walk_forward(fold_metrics, primary_metric="qlike")
    ridge_qlike = float(summary.loc[summary["model"] == "ridge", "qlike"].iloc[0])
    mean_qlike = float(
        summary.loc[summary["model"] == "historical_mean", "qlike"].iloc[0]
    )
    assert ridge_qlike < mean_qlike


def test_random_forest_beats_mean_on_walk_forward() -> None:
    """Random forest should beat historical mean on a nonlinear target."""
    features, target = _nonlinear_design()
    registry = create_default_model_registry()
    models = registry.create_many(["historical_mean", "random_forest"])

    fold_metrics = run_walk_forward(
        features=features,
        target=target,
        models=models,
        n_splits=N_SPLITS,
        embargo_size=EMBARGO_SIZE,
    )
    summary = summarize_walk_forward(fold_metrics, primary_metric="qlike")
    forest_qlike = float(
        summary.loc[summary["model"] == "random_forest", "qlike"].iloc[0]
    )
    mean_qlike = float(
        summary.loc[summary["model"] == "historical_mean", "qlike"].iloc[0]
    )
    assert forest_qlike < mean_qlike