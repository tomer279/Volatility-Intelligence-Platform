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

## Factor importance (Milestone 4/5)

- Primary screening model: **Ridge**
- Importance = ΔQLIKE after shuffling one **test** feature column
- Model is not refit inside a permutation
- Fold-level importance = mean over shuffle repeats (optional `delta_cap`)
- Ranking aggregate: **median** across folds (mean/std also reported); top-k hit rate across folds

## Cross-asset features (Milestone 5)

- Optional VIX covariates: `vix_level`, `vix_chg_1d`
- Storage symbol: `VIX` (Yahoo ticker `^VIX` via ingest alias)
- Alignment: backward as-of join onto the primary session calendar (`timestamp ≤ t`)
- Enable with `vip features --symbol SPY --with-vix` after `vip ingest --symbol VIX`
- Associative only: VIX is contemporaneously correlated with equity vol; not a causal claim

## Regime slices (Milestone 5)

- `full_sample`: all walk-forward test rows
- `covid_crash`: 2020-02-20 .. 2020-04-30
- `bear_2022`: 2022-01-03 .. 2022-10-14
- Metrics recomputed on pooled OOS predictions inside each window
- Empty windows reported with `n_obs=0` (no crash)

## Nonlinear attribution (Milestone 5)

- Random forest walk-forward horse-race candidate (wired fully in screen Step 7)
- TreeSHAP: fit on train, explain **test** features only; importance = mean |SHAP|
- Aggregate SHAP ranks with median across folds
- Optional dependency: `shap` via `pip install -e ".[nonlinear]"`
- Complementary to Ridge permutation ΔQLIKE; not causal

## Tree models (Milestone 5)

- RandomForest (`n_estimators=200`, `max_depth=4`, `min_samples_leaf=5`)
- No feature scaling (trees are scale-invariant)
- Predictions floored at `1e-8` for QLIKE stability
- Walk-forward horse-race alongside HAR-RV OLS, Ridge, Lasso

## Reporting

- `vip screen` writes JSON artifacts, `importance_plot.png`, and `report.html`
- HTML memo includes methodology, horse-race, ranked factors, "What works when" regime table, plot, and caveats
- `vip screen-batch` loops over multiple symbols with caching (ingest/features only when missing)

## Caveats

- HAR lags are collinear; treat them as a family, not independent discoveries
- Permutation importance is associative, not causal
- Weak-factor ranks can be unstable across folds/regimes
- MVP results are for SPY; do not over-generalize without multi-symbol checks
- VIX features are as-of joined; do not interpret as exogenous shocks without further design
- QLIKE permutation deltas can spike on collinear HAR lags; rankings use median importance across folds
- Multi-symbol batch results share methodology but each symbol's regime coverage may differ