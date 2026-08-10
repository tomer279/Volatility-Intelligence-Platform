# `vip.modeling`

## Purpose
Implement volatility forecasting models behind a shared fit/predict interface.

## Modules
- `baselines.py` - Historical mean, EWMA, HAR-RV OLS, and VIX-as-forecast baselines.
- `regularization.py` - Ridge, Lasso, and ElasticNet with train-only scaling.
- `registry.py` - Name → factory registry for baselines, linear, and tree models.
- `tree_models.py` - Random forest (unscaled); LightGBM optional (`.[nonlinear]`)

## Key APIs
- `HistoricalMeanModel` - Constant forecast equal to the training-target mean.
- `EwmaModel` - Frozen end-of-train EWMA level forecast.
- `HarRvOlsModel` - OLS on `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d` (+ intercept).
- `RidgeModel` / `LassoModel` / `ElasticNetModel` - Scaled linear models on all provided features.
- `VixAsForecastModel` (`vix_as_forecast`) - Intercept OLS on `vix_vol_daily`
(or `vix_level` via locked conversion).
- `create_default_model_registry()` - Baselines (incl. `vix_as_forecast`) +
ridge/lasso/elasticnet + random_forest.
- `ModelRegistry.create_many(names)` - Build a dict for `run_walk_forward`.
- `RandomForestVolModel` - Random forest on all provided features (no scaling).

## Notes
- All models expose `fit(features, target)` / `predict(features)` for evaluator uniformity.
- Regularized models fit `StandardScaler` inside `fit` only (per fold when used in walk-forward).
- Predictions are floored at `1e-8` to keep QLIKE stable.
- Fit only on training rows; never use test labels inside `fit`.
- Keep vendor/data I/O out of this package.
- Tree models intentionally skip `StandardScaler`; floors still apply at `1e-8`.
- Research roles (M9): `VixAsForecastModel` is a **competing forecast**
  (univariate OLS on daily VIX vol). IV−RV **gap** columns are screened as
  features elsewhere (Ridge/Lasso), not inside this baseline.
- Registry name: `vix_as_forecast`. Prefer `vix_vol_daily` when present;
  otherwise derive from `vix_level` via `vix_level_to_daily_vol`.