# `vip.evaluation`

## Purpose
Score volatility forecasts and run time-safe walk-forward evaluation.

## Modules
- `metrics.py` - QLIKE, MSE, MAE.
- `splitting.py` - Expanding walk-forward fold generation with embargo.
- `walk_forward.py` - Fit/predict/score loop across folds.
- `comparison.py` - Aggregate comparison tables.
- `importance.py` - Walk-forward permutation importance under QLIKE.
- `stability.py` - Mean/std importance and top-k hit rate across folds.

## Key APIs
- `mse(y_true, y_pred)` - Mean squared error.
- `mae(y_true, y_pred)` - Mean absolute error.
- `qlike(y_true, y_pred, epsilon=1e-8)` - QLIKE loss (lower is better).
- `WalkForwardFold` - One fold with `train_index` / `test_index`.
- `generate_expanding_folds(index, n_splits, embargo_size)` - Build chronological expanding folds.
- `run_walk_forward(features, target, models, n_splits, embargo_size)` - Per-fold metric table.
- `summarize_walk_forward(fold_metrics, primary_metric='qlike')` - Mean metrics by model, sorted best-first.
- `permutation_importance_folds(features, target, model_factory, fold_spec, options=None)` - ΔQLIKE importance by fold/feature.
- `WalkForwardSpec` / `ImportanceOptions` - Nested settings (keeps call sites ≤5 params).
- `summarize_importance(importance, options=None)` - Ranked factor stability table.
- `StabilityOptions` - Top-k settings for hit-rate stability.

## Research defaults
- Primary metric: QLIKE (lower is better)
- Secondary metrics: MSE, MAE
- Embargo: at least the forecast horizon (5 trading days)
- Split style: expanding training window

## Notes
- Predictions are clipped by a small epsilon inside QLIKE to avoid non-positive forecasts.
- Train indices always end before the embargo; test indices never overlap train.
- Models are refit on every fold; do not reuse test labels inside `fit`.
- Permutation importance shuffles **test** columns only; the model is not refit inside a shuffle.
- Importance = permuted QLIKE − baseline QLIKE (higher means more important).
