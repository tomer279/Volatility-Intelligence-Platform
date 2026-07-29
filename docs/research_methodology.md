# Research Methodology (Draft)

## Target

- Forward 5-trading-day close-to-close realized volatility: `target_rv_cc_5d`
- Stored non-annualized

## Primary metric

- QLIKE (lower is better)
- Secondary: MSE, MAE

## Validation

- Expanding walk-forward
- Embargo ≥ 5 trading days between train and test
- Models refit each fold using training data only
- Feature scaling (when used) is fit on the training fold only

## Baselines (Milestone 3)

- Historical mean
- EWMA (frozen at end of train)
- HAR-RV OLS on trailing RV features (`rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d`)

## Regularized models (Milestone 4)

- Ridge, Lasso, ElasticNet via scikit-learn
- `StandardScaler` fit inside each model's `fit` (train window only)
- Predictions floored at `1e-8` for QLIKE stability
- Horse-race vs HAR-RV OLS uses the same walk-forward folds

## Factor importance (Milestone 4)

- Primary screening model: **Ridge**
- Importance = mean ΔQLIKE after shuffling one **test** feature column
- Model is not refit inside a permutation
- Stability summary: mean/std importance and top-k hit rate across folds

## Reporting

- `vip screen` writes JSON artifacts, `importance_plot.png`, and `report.html`
- HTML memo includes methodology, horse-race, ranked factors, plot, and caveats

## Caveats

- HAR lags are collinear; treat them as a family, not independent discoveries
- Permutation importance is associative, not causal
- Weak-factor ranks can be unstable across folds/regimes
- MVP results are for SPY; do not over-generalize without multi-symbol checks