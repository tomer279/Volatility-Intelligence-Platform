"""Tests for walk-forward TreeSHAP importance (optional shap dependency)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")

from vip.evaluation.importance import WalkForwardSpec
from vip.evaluation.shap_importance import shap_available, shap_importance_folds
from vip.evaluation.stability import StabilityOptions, summarize_importance
from vip.modeling.tree_models import RandomForestVolModel

SIGNAL_COLUMN = "signal"
NOISE_COLUMN = "noise"
N_ROWS = 180


def _nonlinear_design() -> tuple[pd.DataFrame, pd.Series]:
    """Build a nonlinear signal design for tree SHAP."""
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 1.0, N_ROWS)
    features = pd.DataFrame(
        {
            SIGNAL_COLUMN: signal,
            NOISE_COLUMN: rng.normal(0.0, 1.0, N_ROWS),
        },
        index=index,
    )
    nonlinear = np.where(signal > 0.0, signal**2, 0.1)
    target = pd.Series(
        0.05 + 0.4 * nonlinear + rng.normal(0.0, 0.01, N_ROWS),
        index=index,
        name="target_rv_cc_5d",
    )
    return features, target


def test_shap_available_when_installed() -> None:
    """With shap installed, the availability helper should be True."""
    assert shap_available() is True


def test_shap_signal_ranks_above_noise() -> None:
    """Median SHAP importance of the true signal should exceed noise."""
    features, target = _nonlinear_design()
    importance = shap_importance_folds(
        features=features,
        target=target,
        model_factory=RandomForestVolModel,
        fold_spec=WalkForwardSpec(n_splits=3, embargo_size=5),
    )
    ranking = summarize_importance(
        importance,
        options=StabilityOptions(top_k=1, rank_by="median"),
    )
    assert ranking.iloc[0]["feature"] == SIGNAL_COLUMN


def test_shap_schema_and_no_mutation() -> None:
    """SHAP output schema should be fold/feature/importance; inputs unchanged."""
    features, target = _nonlinear_design()
    before = features.copy(deep=True)
    importance = shap_importance_folds(
        features=features,
        target=target,
        model_factory=RandomForestVolModel,
        fold_spec=WalkForwardSpec(n_splits=3, embargo_size=5),
    )
    assert set(importance.columns) == {"fold_id", "feature", "importance"}
    assert set(importance["feature"]) == {SIGNAL_COLUMN, NOISE_COLUMN}
    pd.testing.assert_frame_equal(features, before)