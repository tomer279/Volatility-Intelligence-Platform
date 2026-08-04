# `vip.evaluation`

## Purpose
Score volatility forecasts and run time-safe walk-forward evaluation,
including statistical inference on out-of-sample QLIKE gaps vs HAR.

## Modules
- `metrics.py` - QLIKE (scalar + per-row), MSE, MAE.
- `splitting.py` - Expanding walk-forward fold generation with embargo.
- `walk_forward.py` - Fit/predict/score; OOS prediction + per-row QLIKE loss panels.
- `comparison.py` - Aggregate comparison tables; inference-enriched horse-race;
  optional non-overlap sensitivity summary.
- `inference.py` - Loss differentials, block bootstrap (primary), optional HLN–DM,
  non-overlapping subsample helpers.
- `importance.py` - Walk-forward permutation importance under QLIKE.
- `stability.py` - Mean/median importance and top-k hit rate across folds.
- `regimes.py` - Locked COVID/2022 windows and regime-sliced OOS metrics.
- `shap_importance.py` - Optional TreeSHAP importance (RF; requires `shap`).
- `horizon_defaults.py` - Embargo / bootstrap-block defaults + validation for h∈{1,5,21}.

## Key APIs (walk-forward / importance)
- `mse(y_true, y_pred)` - Mean squared error.
- `mae(y_true, y_pred)` - Mean absolute error.
- `qlike(y_true, y_pred, epsilon=1e-8)` - QLIKE loss (lower is better).
- `WalkForwardFold` - One fold with `train_index` / `test_index`.
- `generate_expanding_folds(index, n_splits, embargo_size)` - Chronological expanding folds.
- `run_walk_forward(features, target, models, n_splits, embargo_size)` - Per-fold metric table.
- `summarize_walk_forward(fold_metrics, primary_metric='qlike')` - Mean metrics by model, best-first.
- `collect_walk_forward_predictions(...)` - Dated OOS prediction panel.
- `permutation_importance_folds(...)` - ΔQLIKE importance by fold/feature.
- `WalkForwardSpec` / `ImportanceOptions` - Nested settings (keeps call sites ≤5 params).
- `summarize_importance(importance, options=None)` - Ranked factor stability table.
- `StabilityOptions` - Top-k settings for hit-rate stability.
- `score_predictions_by_regime(predictions)` - QLIKE/MSE/MAE by regime × model.
- `shap_importance_folds(...)` - mean |SHAP| on test rows per fold (train-only fits).

## Key APIs (inference)
- `qlike_losses(y_true, y_pred)` - Per-row QLIKE; mean matches `qlike`.
- `attach_qlike_losses(...)` / `collect_walk_forward_oos_losses(...)` - Dated OOS loss panel.
- `loss_differential(challenger, baseline)` - Aligned `d_t = L_challenger - L_baseline`.
- `BootstrapInferenceOptions` / `BootstrapResult` / `DMResult` - Inference settings and results.
- `InferenceSummaryOptions` - Baseline / horizon / bootstrap / HLN settings for enrichment.
- `block_bootstrap_mean(differential, options=None)` - **Primary** inference.
- `hln_diebold_mariano(differential, nw_lags, horizon_days=None)` - Secondary HLN–DM.
- `nw_lags_for_horizon(horizon_days)` - Locked to `horizon_days - 1`.
- `summarize_with_inference(fold_metrics, oos_losses, options=None)` - Horse-race + CI/p.
- `non_overlapping_index(index, horizon_days)` - Every-`horizon_days`-th OOS label.
- `non_overlapping_subsample(series, horizon_days)` - Footnote thinning of a series.
- `block_bootstrap_nonoverlap_sensitivity(...)` - Bootstrap on the thinned differential.
- `NonOverlapSensitivityResult` - JSON-friendly footnote row payload.
- `summarize_nonoverlap_sensitivity(oos_losses, options=None)` - Footnote bootstrap table.
- `default_embargo_for_horizon(horizon_days)` - Embargo = horizon.
- `default_bootstrap_block_length(horizon_days)` - Locked ℓ for h∈{1,5,21}.
- `allowed_bootstrap_block_range(horizon_days)` / `validate_bootstrap_block_length(...)`.
- `BootstrapBlockBounds` - Pass into `BootstrapInferenceOptions` for h≠5 ranges.

## Research defaults (M7 / M8)
- Primary inference: block bootstrap of mean OOS ΔQLIKE vs `har_rv_ols`
  (B=1999; α=0.05; seed 0).
- Horizon-aware block defaults (M8): ℓ=**10 / 15 / 21** with ranges
  **5–15 / 10–20 / 15–42** for h=**1 / 5 / 21**
  (`default_bootstrap_block_length`, `BootstrapBlockBounds` via
  `settings_for_horizon`). Legacy single-horizon path remains h=5 → ℓ=15.
- Embargo for multi-horizon screens: `default_embargo_for_horizon(h)` → `h`.
- Secondary: HLN–DM with NW lags = `nw_lags_for_horizon(h)` (= `h − 1`).
- “Significantly better” only when primary bootstrap rejects at α and ΔQLIKE < 0.
- Non-overlapping every-horizon subsample is a footnote only.
- Embargo blocks train/test leakage; it is not a significance test.
- Multi-horizon **orchestration** lives in `vip.application.screen_multi_horizon`
  (`vip screen-horizons`); this package owns metrics, splits, and inference math.

## Notes
- Predictions are clipped by a small epsilon inside QLIKE to avoid non-positive forecasts.
- Train indices always end before the embargo; test indices never overlap train.
- Models are refit on every fold; do not reuse test labels inside `fit`.
- Inference uses per-row OOS losses only; never refit models inside bootstrap replicates.
- Permutation importance shuffles **test** columns only; the model is not refit inside a shuffle.
- Importance = permuted QLIKE − baseline QLIKE (higher means more important).
- Ranking aggregate defaults to **median** ΔQLIKE across folds (mean still reported); optional `delta_cap` clips per-shuffle deltas.
- Install optional TreeSHAP deps: `pip install -e ".[nonlinear]"`.