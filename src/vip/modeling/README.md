# `vip.modeling`

## Purpose
Implement volatility forecasting models behind a shared fit/predict interface.

## Modules
- `baselines.py` - Historical mean, EWMA, and HAR-RV OLS baselines.
- `regularization.py` - Ridge, Lasso, and ElasticNet with train-only scaling.
- `registry.py` - Name → factory registry for baselines and regularized models.

## Key APIs
- `HistoricalMeanModel` - Constant forecast equal to the training-target mean.
- `EwmaModel` - Frozen end-of-train EWMA level forecast.
- `HarRvOlsModel` - OLS on `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d` (+ intercept).
- `RidgeModel` / `LassoModel` / `ElasticNetModel` - Scaled linear models on all provided features.
- `create_default_model_registry()` - Baselines + ridge/lasso/elasticnet factories.
- `ModelRegistry.create_many(names)` - Build a dict for `run_walk_forward`.

## Notes
- All models expose `fit(features, target)` / `predict(features)` for evaluator uniformity.
- Regularized models fit `StandardScaler` inside `fit` only (per fold when used in walk-forward).
- Predictions are floored at `1e-8` to keep QLIKE stable.
- Fit only on training rows; never use test labels inside `fit`.
- Keep vendor/data I/O out of this package.