# `vip.modeling`

## Purpose
Implement volatility forecasting models behind a shared fit/predict interface.

## Modules
- `baselines.py` - Historical mean, EWMA, and HAR-RV OLS baselines.
- `registry.py` - Optional model registry (later milestones).

## Key APIs
- `HistoricalMeanModel` - Constant forecast equal to the training-target mean.
- `EwmaModel` - Frozen end-of-train EWMA level forecast.
- `HarRvOlsModel` - OLS on `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d` (+ intercept).

## Notes
- All models expose `fit(features, target)` / `predict(features)` for evaluator uniformity.
- Historical mean and EWMA ignore `features` inside `fit`.
- Fit only on training rows; never use test labels inside `fit`.
- Keep vendor/data I/O out of this package.