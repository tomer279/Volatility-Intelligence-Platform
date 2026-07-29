"""Forecast evaluation metrics and walk-forward utilities.

Exports
-------
mae
    Mean absolute error.
mse
    Mean squared error.
qlike
    QLIKE volatility forecast loss.
WalkForwardFold
    One train/test fold with identifiers.
generate_expanding_folds
    Build expanding-window folds with an embargo gap.
run_walk_forward
    Fit/predict/score models across expanding folds.
summarize_walk_forward
    Aggregate fold metrics by model and sort by primary metric.
WalkForwardSpec
    Expanding walk-forward settings for importance runs.
ImportanceOptions
    Repeat count and RNG seed for column shuffles.
permutation_importance_folds
    Fold-wise permutation importance under QLIKE.
"""

from vip.evaluation.comparison import summarize_walk_forward
from vip.evaluation.importance import (
    ImportanceOptions,
    WalkForwardSpec,
    permutation_importance_folds,
)
from vip.evaluation.metrics import mae, mse, qlike
from vip.evaluation.splitting import WalkForwardFold, generate_expanding_folds
from vip.evaluation.walk_forward import run_walk_forward
from vip.evaluation.stability import StabilityOptions, summarize_importance

__all__ = [
    "ImportanceOptions",
    "WalkForwardFold",
    "WalkForwardSpec",
    "generate_expanding_folds",
    "mae",
    "mse",
    "permutation_importance_folds",
    "qlike",
    "run_walk_forward",
    "summarize_walk_forward",
    "StabilityOptions",
    "summarize_importance",
]