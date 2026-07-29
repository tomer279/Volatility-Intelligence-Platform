"""Tests for walk-forward permutation importance under QLIKE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vip.evaluation.importance import (
    ImportanceOptions,
    WalkForwardSpec,
    permutation_importance_folds,
)
from vip.modeling.regularization import RidgeModel

N_ROWS = 240
SIGNAL_COLUMN = "signal"
NOISE_COLUMN = "noise"


def _synthetic_signal_design() -> tuple[pd.DataFrame, pd.Series]:
    """Build a design where one column drives the target."""
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    rng = np.random.default_rng(7)
    signal = rng.normal(0.0, 1.0, N_ROWS)
    noise = rng.normal(0.0, 1.0, N_ROWS)
    features = pd.DataFrame(
        {
            SIGNAL_COLUMN: signal,
            NOISE_COLUMN: noise,
        },
        index=index,
    )
    target = pd.Series(
        2.0 + 0.8 * signal + rng.normal(0.0, 0.02, N_ROWS),
        index=index,
        name="target_rv_cc_5d",
    )
    return features, target


def test_signal_ranks_above_noise() -> None:
    """Mean importance of the true signal should exceed pure noise."""
    features, target = _synthetic_signal_design()
    importance = permutation_importance_folds(
        features=features,
        target=target,
        model_factory=RidgeModel,
        fold_spec=WalkForwardSpec(n_splits=4, embargo_size=5),
        options=ImportanceOptions(n_repeats=3, random_seed=0),
    )
    means = importance.groupby("feature")["importance"].mean()
    assert float(means[SIGNAL_COLUMN]) > float(means[NOISE_COLUMN])


def test_permutation_does_not_mutate_features() -> None:
    """Column shuffles must not mutate the caller's feature matrix."""
    features, target = _synthetic_signal_design()
    before = features.copy(deep=True)
    permutation_importance_folds(
        features=features,
        target=target,
        model_factory=RidgeModel,
        fold_spec=WalkForwardSpec(n_splits=3, embargo_size=5),
        options=ImportanceOptions(n_repeats=2, random_seed=1),
    )
    pd.testing.assert_frame_equal(features, before)


def test_importance_schema() -> None:
    """Output should include fold/feature importance columns."""
    features, target = _synthetic_signal_design()
    importance = permutation_importance_folds(
        features=features,
        target=target,
        model_factory=RidgeModel,
        fold_spec=WalkForwardSpec(n_splits=3, embargo_size=5),
        options=ImportanceOptions(n_repeats=2, random_seed=2),
    )
    assert set(importance.columns) == {
        "fold_id",
        "feature",
        "importance",
        "baseline_qlike",
        "n_repeats",
    }
    assert set(importance["feature"]) == {SIGNAL_COLUMN, NOISE_COLUMN}
    assert importance["fold_id"].nunique() == 3