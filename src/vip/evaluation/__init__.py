"""Forecast evaluation metrics and walk-forward utilities.

Exports
-------
mae
    Mean absolute error.
mse
    Mean squared error.
qlike
    QLIKE volatility forecast loss.
qlike_losses
    Per-row QLIKE losses for volatility forecasts.
WalkForwardFold
    One train/test fold with identifiers.
generate_expanding_folds
    Build expanding-window folds with an embargo gap.
run_walk_forward
    Fit/predict/score models across expanding folds.
collect_walk_forward_predictions
    Collect dated out-of-sample predictions across expanding folds.
attach_qlike_losses
    Add per-row QLIKE losses to a walk-forward prediction panel.
collect_walk_forward_oos_losses
    Collect per-row OOS QLIKE losses across expanding folds.
summarize_walk_forward
    Aggregate fold metrics by model and sort by primary metric.
InferenceSummaryOptions
    Baseline, bootstrap, HLN–DM, and horizon settings for enrichment.
summarize_with_inference
    Horse-race means plus bootstrap (and optional HLN–DM) vs baseline.
summarize_nonoverlap_sensitivity
    Footnote bootstrap on every-horizon-day OOS differentials.
WalkForwardSpec
    Expanding walk-forward settings for importance runs.
ImportanceOptions
    Repeat count and RNG seed for column shuffles.
permutation_importance_folds
    Fold-wise permutation importance under QLIKE.
StabilityOptions
    Top-k settings for importance hit-rate stability.
summarize_importance
    Ranked factor stability table across folds.
BootstrapInferenceOptions
    Block-bootstrap settings for mean loss differentials.
BootstrapResult
    Mean gap, percentile CI, and two-sided bootstrap p-value.
DMResult
    Diebold–Mariano statistic with HLN correction and p-value.
NonOverlapSensitivityResult
    Footnote bootstrap result on a horizon-strided subsample.
nw_lags_for_horizon
    Newey–West lag locked to ``horizon_days - 1``.
loss_differential
    Aligned challenger minus baseline per-row losses.
block_bootstrap_mean
    Moving block bootstrap of the mean loss differential.
hln_diebold_mariano
    DM test with Newey–West HAC and HLN finite-sample correction.
non_overlapping_index
    Keep every horizon_days-th label from a sorted unique index.
non_overlapping_subsample
    Thin a time-ordered series to non-overlapping horizon spacing.
block_bootstrap_nonoverlap_sensitivity
    Block-bootstrap mean(d) on the non-overlapping subsample.
LOCKED_SCREEN_HORIZONS
    Locked multi-horizon study set ``(1, 5, 21)``.
default_embargo_for_horizon
    Embargo size locked to ``horizon_days``.
default_bootstrap_block_length
    Locked default bootstrap block length for a horizon.
allowed_bootstrap_block_range
    Inclusive allowed block-length interval for a horizon.
validate_bootstrap_block_length
    Horizon-aware block-length validator.
BootstrapBlockBounds
    Inclusive allowed interval for ``BootstrapInferenceOptions.block_length``.
"""
from vip.evaluation.horizon_defaults import (
    LOCKED_SCREEN_HORIZONS,
    allowed_bootstrap_block_range,
    default_bootstrap_block_length,
    default_embargo_for_horizon,
    validate_bootstrap_block_length,
)
from vip.evaluation.comparison import (
    InferenceSummaryOptions,
    summarize_nonoverlap_sensitivity,
    summarize_walk_forward,
    summarize_with_inference,
)
from vip.evaluation.importance import (
    ImportanceOptions,
    WalkForwardSpec,
    permutation_importance_folds,
    DEFAULT_IMPORTANCE_DELTA_CAP
)
from vip.evaluation.inference import (
    BootstrapBlockBounds,
    BootstrapInferenceOptions,
    BootstrapResult,
    DMResult,
    NonOverlapSensitivityResult,
    block_bootstrap_mean,
    block_bootstrap_nonoverlap_sensitivity,
    hln_diebold_mariano,
    loss_differential,
    non_overlapping_index,
    non_overlapping_subsample,
    nw_lags_for_horizon,
)
from vip.evaluation.metrics import mae, mse, qlike, qlike_losses
from vip.evaluation.splitting import WalkForwardFold, generate_expanding_folds
from vip.evaluation.stability import StabilityOptions, summarize_importance
from vip.evaluation.walk_forward import (
    attach_qlike_losses,
    collect_walk_forward_oos_losses,
    collect_walk_forward_predictions,
    run_walk_forward,
)

__all__ = [
    "ImportanceOptions",
    "WalkForwardFold",
    "WalkForwardSpec",
    "generate_expanding_folds",
    "mae",
    "mse",
    "permutation_importance_folds",
    "qlike",
    "qlike_losses",
    "run_walk_forward",
    "summarize_walk_forward",
    "StabilityOptions",
    "summarize_importance",
    "BootstrapInferenceOptions",
    "BootstrapResult",
    "DMResult",
    "NonOverlapSensitivityResult",
    "block_bootstrap_mean",
    "block_bootstrap_nonoverlap_sensitivity",
    "hln_diebold_mariano",
    "loss_differential",
    "non_overlapping_index",
    "non_overlapping_subsample",
    "nw_lags_for_horizon",
    "attach_qlike_losses",
    "collect_walk_forward_oos_losses",
    "collect_walk_forward_predictions",
    "InferenceSummaryOptions",
    "summarize_with_inference",
    "summarize_nonoverlap_sensitivity",
    "LOCKED_SCREEN_HORIZONS",
    "BootstrapBlockBounds",
    "allowed_bootstrap_block_range",
    "default_bootstrap_block_length",
    "default_embargo_for_horizon",
    "validate_bootstrap_block_length",
    "DEFAULT_IMPORTANCE_DELTA_CAP"
]
