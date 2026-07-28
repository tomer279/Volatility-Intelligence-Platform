# `vip.evaluation`

## Purpose
Score volatility forecasts and run time-safe walk-forward evaluation.

## Modules
- `metrics.py` - QLIKE, MSE, MAE.
- `splitting.py` - Expanding walk-forward fold generation with embargo.
- `walk_forward.py` - Fit/predict/score loop across folds.
- `comparison.py` - Aggregate comparison tables.

## Key APIs
- `mse(y_true, y_pred)` - Mean squared error.
- `mae(y_true, y_pred)` - Mean absolute error.
- `qlike(y_true, y_pred, epsilon=1e-8)` - QLIKE loss (lower is better).
- `WalkForwardFold` - One fold with `train_index` / `test_index`.
- `generate_expanding_folds(index, n_splits, embargo_size)` - Build chronological expanding folds.
- `run_walk_forward(features, target, models, n_splits, embargo_size)` - Per-fold metric table.
- `summarize_walk_forward(fold_metrics, primary_metric='qlike')` - Mean metrics by model, sorted best-first.

## Research defaults
- Primary metric: QLIKE (lower is better)
- Secondary metrics: MSE, MAE
- Embargo: at least the forecast horizon (5 trading days)
- Split style: expanding training window

## Notes
- Predictions are clipped by a small epsilon inside QLIKE to avoid non-positive forecasts.
- Train indices always end before the embargo; test indices never overlap train.
- Models are refit on every fold; do not reuse test labels inside `fit`.