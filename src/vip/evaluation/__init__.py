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
"""

from vip.evaluation.comparison import summarize_walk_forward
from vip.evaluation.metrics import mae, mse, qlike
from vip.evaluation.splitting import WalkForwardFold, generate_expanding_folds
from vip.evaluation.walk_forward import run_walk_forward

__all__ = [
    "WalkForwardFold",
    "generate_expanding_folds",
    "mae",
    "mse",
    "qlike",
    "run_walk_forward",
    "summarize_walk_forward",
]