# `vip.modeling`

## Purpose
Implement volatility forecasting models behind a shared fit/predict interface.

## Modules
- `baselines.py` - Historical mean, EWMA, HAR-RV OLS, VIX-as-forecast, and
  discrete OU (`ou_rv`) baselines.
- `parametric.py` - Stretch recursive EWMA filter (`ewma_recursive`).
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
- `OuRvModel` (`ou_rv`) - Frozen-origin discrete OU / AR(1) on log target;
  analytic h-step mean; default `horizon_days=5` (screens inject h for M8).
- `create_default_model_registry()` - Baselines (incl. `vix_as_forecast`,
  `ou_rv`, `ewma_recursive`) + ridge/lasso/elasticnet + random_forest.
- `ModelRegistry.create_many(names)` - Build a dict for `run_walk_forward`.
- `RandomForestVolModel` - Random forest on all provided features (no scaling).
- `EwmaRecursiveModel` (`ewma_recursive`) - Train-fit EWMA decay; recursive
  OOS updates via trailing `rv_cc_1d` (stretch; ≠ frozen `ewma`).

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
- Registry name: `ou_rv`. Univariate on the training target; always eligible
  for the factor-screen horse-race (no VIX column gate). Default factory is
  zero-arg (h=5); `resolve_horse_race_models` passes `horizon_days` when h≠5.
- Registry name: `ewma_recursive`. Train-fit decay; recursive OOS updates via
  trailing `rv_cc_1d` (not the forward label). Always in the horse-race; distinct
  from frozen `ewma`.
- Research contract (M10): discrete OU on log target; frozen-origin h-step
  mean; physical measure; must beat `har_rv_ols` under M7 bootstrap to matter
  — see `docs/research_methodology.md` §13.