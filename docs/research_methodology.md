# Research Methodology

This document describes the quantitative methodology used across milestones 1–7
of the Volatility Intelligence Platform.  It is written for a technically
literate reader (quant analyst, PM, or researcher) and references the locked
defaults from `plan.md`.

---

## 1  Target Variable


| Setting       | Value                                                    |
| ------------- | -------------------------------------------------------- |
| Name          | `target_rv_cc_5d`                                        |
| Definition    | Forward 5-trading-day close-to-close realized volatility |
| Annualization | None (stored raw)                                        |


At each date *t* the target is computed from log returns over (t, t + 5]:

```
target_rv_cc_5d(t) = sqrt( sum_{i=1}^{5} r_{t+i}^2 )
```

where `r_{t+i} = log(close_{t+i} / close_{t+i-1})`.

---

## 2  Feature Families (Milestone 2)

All features at date *t* use data up to and including *t* — never future data.
NaN rows created by trailing windows are dropped before modelling.

### 2.1  Returns


| Column   | Description                                     |
| -------- | ----------------------------------------------- |
| `ret_1d` | One-day log return `log(close_t / close_{t-1})` |
| `ret_5d` | Cumulative 5-day log return                     |


### 2.2  HAR trailing realized volatility


| Column      | Description                       |
| ----------- | --------------------------------- |
| `rv_cc_1d`  | 1-day trailing close-to-close RV  |
| `rv_cc_5d`  | 5-day trailing close-to-close RV  |
| `rv_cc_21d` | 21-day trailing close-to-close RV |


These three lags form the heterogeneous autoregressive (HAR) family.  They are
collinear by construction; treat them as a single family when interpreting
importance rankings (see §10 Caveats).

### 2.3  Range


| Column          | Description                      |
| --------------- | -------------------------------- |
| `range_1d`      | Daily `log(high / low)`          |
| `range_5d_mean` | 5-day rolling mean of `range_1d` |


### 2.4  Volume


| Column         | Description                      |
| -------------- | -------------------------------- |
| `volume_z_21d` | 21-day z-score of trading volume |


### 2.5  Cross-asset — VIX (Milestone 5)


| Column       | Description                               |
| ------------ | ----------------------------------------- |
| `vix_level`  | VIX closing level (backward as-of joined) |
| `vix_chg_1d` | One-day change in VIX level               |


VIX data is ingested separately (`vip ingest --symbol VIX`) and aligned to the
primary symbol's trading calendar via a **backward as-of join**
(`timestamp ≤ t`).  This guarantees no forward-fill leakage: if VIX has no
observation on date *t*, the most recent prior value is used.

VIX features are enabled with the `--with-vix` flag.  They are associative
(contemporaneously correlated with equity vol), not causal — see §10.

---

## 3  Primary Metric


| Metric | Formula | Direction |
| ------ | ------- | --------- |
| QLIKE  | `mean( log(ŷ²) + y² / ŷ² )` | Lower is better |
| MSE    | `mean( (y - ŷ)² )` | Lower is better |
| MAE    | `mean(abs(y - ŷ))` | Lower is better |


QLIKE is the primary metric because it penalises proportional forecast errors
more heavily than MSE, which is appropriate for volatility where the scale varies
across regimes.

All predictions are floored at `ε = 1e-8` before QLIKE evaluation to avoid
log-of-zero instability.

---

## 4  Walk-Forward Validation


| Setting          | Value                                             |
| ---------------- | ------------------------------------------------- |
| Mode             | Expanding window                                  |
| Number of splits | 5                                                 |
| Embargo          | ≥ 5 trading days between train end and test start |


Models are refit on each training fold.  Feature scaling (when used) is fit on
the training fold only and applied to both train and test — never on the full
sample.

---

## 5  Baselines (Milestone 3)

Three baselines establish the forecasting floor.

### 5.1  Historical Mean

Predict the training-set mean of `target_rv_cc_5d` for every test row.  This is
the naïve benchmark.

### 5.2  EWMA

Exponentially-weighted moving average (λ = 0.94) of squared daily log returns,
frozen at the end of the training window and held constant across the test fold.

### 5.3  HAR-RV OLS

OLS regression of `target_rv_cc_5d` on the three HAR trailing RV features
(`rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d`).  No intercept scaling; fit per fold.

**Success criterion (M3):** HAR-RV OLS beats historical mean on QLIKE.

---

## 6  Regularised Linear Models (Milestone 4)


| Model      | Key hyperparameters                |
| ---------- | ---------------------------------- |
| Ridge      | `alpha=1.0` (scikit-learn default) |
| Lasso      | `alpha=1.0`                        |
| ElasticNet | `alpha=1.0, l1_ratio=0.5`          |


Each model wraps a `StandardScaler` that is fit on the training fold only.
Predictions are floored at `1e-8`.  The horse-race comparison uses the same
walk-forward folds as the baselines.

---

## 7  Tree Models (Milestone 5)


| Setting            | Value                            |
| ------------------ | -------------------------------- |
| Algorithm          | `RandomForestRegressor`          |
| `n_estimators`     | 200                              |
| `max_depth`        | 4                                |
| `min_samples_leaf` | 5                                |
| `random_state`     | 0                                |
| Feature scaling    | None (trees are scale-invariant) |


Predictions are floored at `1e-8` for QLIKE stability.  RandomForest
participates in the same walk-forward horse-race alongside HAR-RV OLS, Ridge,
and Lasso.

Because trees are unscaled, they receive the raw feature matrix — no
`StandardScaler` wrapper.

---

## 8  Factor Importance

### 8.1  Permutation importance (Milestone 4)

The primary screening model is **Ridge**.  For each walk-forward fold:

1. Score the test set with the fitted model → baseline QLIKE.
2. For each feature column *j*, shuffle the column in the test set and re-score → QLIKE_shuffled.
3. Importance = ΔQLIKE = QLIKE_shuffled − QLIKE_baseline.

The model is **not** refit inside permutations.  An optional `delta_cap`
truncates extreme ΔQLIKE values within a fold.

### 8.2  Aggregation — median importance (Milestone 5)

Fold-level importance values are aggregated across folds using the **median**
(not mean) of ΔQLIKE.  Median is robust to QLIKE spikes that arise from
collinear HAR lags in individual folds.  Mean and standard deviation are also
reported for reference.

Top-k hit-rate across folds is reported as a secondary stability indicator.

### 8.3  SHAP attribution (Milestone 5)

TreeSHAP provides a complementary, model-specific view of feature importance:

1. A `RandomForestRegressor` is fit on the training fold.
2. `TreeExplainer` explains **test-set** predictions only (no training leakage).
3. Fold importance = mean of absolute SHAP values across test rows.
4. Cross-fold aggregate = **median** of fold-level mean |SHAP|.

SHAP ranks are complementary to Ridge permutation ΔQLIKE.  They capture
nonlinear and interaction effects that linear permutation cannot.  Neither method
makes causal claims.

SHAP is an optional dependency (`pip install -e ".[nonlinear]"`).

---

## 9  Regime-Sliced Evaluation (Milestone 5)


| Regime        | Date range                 |
| ------------- | -------------------------- |
| `full_sample` | All walk-forward test rows |
| `covid_crash` | 2020-02-20 → 2020-04-30    |
| `bear_2022`   | 2022-01-03 → 2022-10-14    |


Metrics are recomputed on the pooled out-of-sample predictions that fall inside
each regime window.  This answers "which features matter in calm vs. crisis
markets?"

**Empty-slice handling:** if no test rows fall within a regime window (e.g. the
data range does not cover the COVID period), the regime row is reported with
`n_obs = 0` and no metrics — the platform does not crash.

---

## 10  Statistical inference on OOS gaps (Milestone 7)

Mean walk-forward QLIKE rankings are **descriptive**. Overlapping multi-day RV
labels (`target_rv_cc_5d`) induce serial dependence in per-row loss
differentials, so a gap such as “Lasso beats HAR by 0.05 QLIKE” is not a
finding until uncertainty is attached.

### 10.1  What embargo does — and does not do

The walk-forward **embargo** (default 5 sessions) blocks train/test leakage
around the forecast horizon. It is **not** a statistical test that a model gap
is real. Inference is a separate step on out-of-sample losses only.

### 10.2  Per-row loss differentials

For each horse-race challenger vs baseline `har_rv_ols`, form

```text
d_t = L_t(challenger) − L_t(baseline)
```

on aligned OOS session dates, where `L_t` is per-row QLIKE. Mean ΔQLIKE =
`mean(d_t)` (negative favors the challenger). Inference is **never** run on the
five fold-mean QLIKE values alone.

### 10.3  Primary inference — block bootstrap

| Setting | Locked default |
| --- | --- |
| Method | Moving block bootstrap of `mean(d_t)` |
| Block length | 15 trading days (allowed range 10–20) |
| Resamples | 1999 (999 acceptable in unit tests) |
| α | 0.05 (two-sided percentile CI) |
| Seed | 0 |

Report mean ΔQLIKE, bootstrap CI, and two-sided bootstrap p-value for
H0: E[`d_t`] = 0. Block (not i.i.d. day) resampling is required because
overlapping 5-day labels correlate nearby `d_t`.

### 10.4  Secondary inference — HLN–DM + Newey–West

When enabled, also report Diebold–Mariano with Newey–West HAC and the
Harvey–Leybourne–Newbold finite-sample correction:

- NW lags = `horizon_days − 1` → **4** for the default 5-day target
- Persist `dm_stat`, `hln_stat`, `hln_pvalue`, `nw_lags` as **secondary** columns
- Do **not** claim “significantly better” from uncorrected DM, or from HLN–DM
  alone when the bootstrap does not reject

### 10.5  Report wording

- “**Significantly** lower mean OOS QLIKE vs HAR” **only** when the **primary
  bootstrap** rejects at α **and** mean ΔQLIKE &lt; 0
- Otherwise: “lower / higher mean OOS QLIKE vs HAR (not significant at α)”
- Fold-mean horse-race tables without inference columns remain descriptive

### 10.6  Optional sensitivity (footnote)

Subsample OOS dates to non-overlapping horizon spacing (every `horizon_days`-th
OOS row; every 5th trading day for h=5). Recompute mean ΔQLIKE ± block bootstrap
on the thinner series and persist as `inference_sensitivity.json`. This is a
**footnote robustness check**, not a second primary test and not a substitute
for the overlapping-sample bootstrap.

### 10.7  Effective sample size (qualitative)

Overlapping h-step labels reduce the effective independent sample relative to
raw OOS row count. The platform does not claim a single scalar “effective N” as
a formal estimator; instead it (a) uses block bootstrap with length in 10–20,
(b) locks NW lag to h−1, and (c) optionally reports the non-overlapping
subsample footnote.


## 11  Caveats

1. **No causal claims.**  All importance measures (permutation ΔQLIKE, SHAP) are
  associative.  A high-ranked feature predicts well; it does not *cause*
   volatility.
2. **HAR lag collinearity.**  `rv_cc_1d`, `rv_cc_5d`, and `rv_cc_21d` are
  mechanically correlated.  Shuffling one while holding the others fixed can
   produce noisy or inflated ΔQLIKE.  Treat the HAR family as one block; median
   aggregation mitigates per-fold spikes.
3. **Regime sensitivity.**  Feature rankings can shift between regimes.  A
  feature that dominates in `full_sample` may be weak during `covid_crash`.
   Always consult the "What Works When" regime table alongside aggregate ranks.
4. **Sample limitations.**  Flagship results use SPY (and optionally QQQ/IWM).
  Do not generalise to other asset classes, geographies, or frequencies without
   additional out-of-sample testing.
5. **VIX contemporaneity.**  VIX features are as-of joined — they reflect
  market-implied vol at the same horizon.  They should not be interpreted as
   exogenous shocks.
6. **QLIKE spike robustness.**  QLIKE is unbounded; a single extreme prediction
  can dominate fold-level importance.  The platform uses median aggregation and
   an optional `delta_cap` to mitigate this.
7. **Walk-forward assumptions.**  Expanding windows assume stationarity of the
  feature–target relationship.  Structural breaks (e.g. post-COVID liquidity
   regime) may violate this.
8. **Inference vs ranking.** Horse-race QLIKE orderings without bootstrap
   inference are descriptive, not findings.
9. **Overlap.** Overlapping RV labels require block bootstrap (and HAC lag
   h−1); i.i.d. day bootstrap understates dependence.