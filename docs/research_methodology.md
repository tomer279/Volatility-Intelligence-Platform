# Research Methodology

This document describes the quantitative methodology used across milestones 1–9
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

VIX features are enabled with ``--with vix`` (or ``--with vix,jump``).
They are associative (contemporaneously correlated with equity vol), not
causal — see §10. For IV−RV gaps and implied-as-forecast, see §2.7 and §12.


### 2.6  Jump-robust daily proxies (Milestone 8 stretch)

| Column | Description |
| ------ | ----------- |
| `jump_prop_1d` / `5d` / `21d` | $\max(0, \mathrm{RV} - \mathrm{BPV}) / \mathrm{RV}$ (0 when RV = 0) |

**Definition (daily proxy).** For close-to-close log returns $r_t$,

$$
\mathrm{BPV}_t(w) = \frac{\pi}{2} \sum |r_i||r_{i-1}|
$$

over adjacent pairs inside the trailing window ending at $t$ (for $w \ge 2$,
$w-1$ pairs among the last $w$ returns; for $w=1$, the single pair
$|r_t||r_{t-1}|$). $\mathrm{RV}$ is the usual sum of squared returns over
the same window. Features are trailing only (information $\le t$).

**Important limitation.** These are **daily close-to-close proxies**, not
Barndorff–Nielsen–Shephard estimators from high-frequency / tick returns.
Do not interpret magnitudes as true jump variation from intraday bipower.
Enable via registry opt-in (`create_default_registry(include_jump=True)`),
`FeatureMatrixExtras(include_jump=True)`, or CLI ``--with jump`` /
``--with vix,jump`` on `vip features`, `vip run`, and `vip screen-horizons`
(rebuilds matrices; ignored when ``--skip-features`` is set unless columns
already exist). Core default families remain returns, har, range, volume
(+ optional VIX).

**Screening contract.** Only `jump_prop_*` columns enter the feature matrix.
Trailing bipower **levels** are not exported as predictors (they are nearly
collinear with `rv_cc_*` and can inflate permutation ΔQLIKE). BPV is still
used internally to define jump proportion.


### 2.7  IV−RV gap family (Milestone 9)

| Column | Description |
| ------ | ----------- |
| `vix_vol_daily` | Locked daily-vol scale of as-of VIX (see conversion below) |
| `vix_minus_rv_1d` / `5d` / `21d` | `vix_vol_daily − rv_cc_{w}d` at HAR windows |
| `vix_rv_ratio_5d` | `vix_vol_daily / rv_cc_5d` (NaN when `rv_cc_5d` ≤ 0) |

**IV proxy.** For liquid index ETFs (flagship SPY), **VIX is the IV proxy**.
VIX is a market-wide implied-vol index, **not** single-name implied volatility
for an individual equity. Do not narrate results as “option IV for SPY” in the
options-surface sense.

**Locked unit conversion.** Platform trailing RV (`rv_cc_*`) is
**non-annualized** close-to-close vol over the trailing window. VIX prints are
conventionally **annualized percent** (e.g. `20` ≈ 20%). All gap features and
`vix_as_forecast` use one conversion:

```text
vix_vol_daily = (vix_level / 100) / sqrt(252)
```

Gaps:

```text
vix_minus_rv_{w}d = vix_vol_daily − rv_cc_{w}d
```

for `w ∈ {1, 5, 21}`. This is a **research proxy** that places both series on
the same daily-vol scale as the target family. It is **not** a variance-swap
replication, options-pricing identity, or claim about fair variance.

**Enablement.** Gaps are opt-in behind CLI ``--with iv_rv`` (implies VIX load)
or `FeatureMatrixExtras(include_iv_rv=True)`. Bare ``--with vix`` still adds only
`vix_level` / `vix_chg_1d`. Builders use as-of VIX (≤ *t*) and trailing
`rv_cc_*` ending at *t*; they do not read `target_rv_cc_*`.

### 2.8  Rates / Treasury yield proxy (Milestone 9 stretch)

Optional cross-asset covariates behind CLI ``--with rates``. Storage symbol
``TNX`` maps to Yahoo ``^TNX`` (10Y yield proxy). No new market-data vendor.

| Column | Description |
| ------ | ----------- |
| `tnx_level` | TNX close (yield percent) as-of session *t* |
| `tnx_chg_1d` | TNX close pct-change as-of *t* (computed on the TNX calendar, then asof-joined) |

**Alignment.** Same backward ``merge_asof`` discipline as VIX: information set
≤ *t* only. Leakage tests mirror the VIX as-of contract.

**Not a vol conversion.** Yield is used as a **level/change covariate**. Do
**not** apply the VIX daily-vol formula
``(level / 100) / sqrt(252)`` to TNX. Rates are not an IV proxy and are not
folded into ``vix_as_forecast``.

**Enablement.** ``vip features --with rates`` / ``FeatureMatrixExtras(include_rates=True)``
after ``vip ingest --symbol TNX`` (or ``vip run --with rates``, which auto-ingests
TNX). Combinable with other tokens, e.g. ``--with vix,iv_rv,rates``.

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
| Embargo          | Default single-horizon path: 5 sessions. Multi-horizon (M8): `embargo_size = h` per horizon (see §11). |


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

### 5.4  VIX as forecast (`vix_as_forecast`, Milestone 9)

Univariate **intercept OLS** of the forward-RV target on `vix_vol_daily`
(derived from `vix_level` via the locked conversion when `vix_vol_daily` is
absent). Predictions floored at `1e-8`. Same fit/predict surface as other
horse-race models.

**Role.** Competing **forecast** of forward RV — not a factor model on the
IV−RV gap vector. Gap columns (`vix_minus_rv_*`) are screened as **features**
(Ridge / Lasso / permutation importance). Keep those research questions
separate in the memo (§12).


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

The model is **not** refit inside permutations.  An optional ``delta_cap``
truncates extreme per-shuffle ΔQLIKE values within a fold.  Factor screens
default to ``delta_cap = 1.0`` (set ``importance_delta_cap=None`` on
``ScreenConfig`` only for diagnostics).

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

For the legacy single-horizon screen (`h = 5`), the table above is the locked
default. Multi-horizon studies use **horizon-aware** block lengths and ranges
(§11); do not force ℓ ∈ [10, 20] when `h = 21`.

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


## 11  Multi-horizon evaluation (Milestone 8)

Milestone 8 promotes forecast horizon from a single config knob to a
**first-class study dimension**. One orchestrated run screens the same
horse-race across locked horizons **1 / 5 / 21** trading days, reusing the
M7 inference stack per horizon (not a parallel CV or significance engine).

Entry points: application `screen_multi_horizon` and CLI `vip screen-horizons`.
Single-horizon `vip screen` / `vip run` remain valid; primary default horizon
stays **5**.

### 11.1  Locked horizons and targets

| Horizon `h` | Target column | Role |
| ----------- | ------------- | ---- |
| 1 | `target_rv_cc_1d` | Next-day forward close-to-close RV |
| 5 | `target_rv_cc_5d` | Next-week (legacy primary) |
| 21 | `target_rv_cc_21d` | Next-month (~21 trading days) |

For each `h`, features are built (or loaded) so the matrix carries
`target_rv_cc_{h}d`. Predictors remain information set ≤ *t*. Walk-forward
is expanding with `n_splits = 5`. Primary metric remains QLIKE; MSE / MAE
are descriptive.

### 11.2  Horizon-scaled validation and inference defaults

Overlapping *h*-step labels induce dependence of order ~*h*. Defaults are
centralized in `vip.evaluation.horizon_defaults` and applied via
`settings_for_horizon(h)`:

| Horizon `h` | `embargo_size` | `nw_lags` (= `h − 1`) | Default `bootstrap_block_length` | Allowed block range |
| ----------- | -------------- | --------------------- | -------------------------------- | ------------------- |
| 1 | 1 | 0 | **10** | 5–15 |
| 5 | 5 | 4 | **15** | 10–20 (M7 unchanged) |
| 21 | 21 | 20 | **21** | 15–42 |

- **Embargo = h** — train/test separation for label overlap; still **not** a
  significance test (§10.1).
- **NW lags = h − 1** — including **0** when `h = 1` on the optional HLN–DM path.
- **Block length** tracks horizon so `h = 21` is not forced into an
  under-blocked ℓ=15; validation must use horizon-specific bounds (legacy
  global [10, 20] alone would reject the h=21 default).

### 11.3  Per-horizon inference (M7 contract carries over)

For each horizon separately:

1. Run the horse-race (`har_rv_ols`, `ridge`, `lasso`, + `random_forest` when
   in the screen) under expanding walk-forward with `embargo_size = h`.
2. Persist per-row OOS QLIKE losses.
3. For each challenger vs baseline **`har_rv_ols`**, compute mean ΔQLIKE and
   **moving block bootstrap** CI / p-value with the horizon’s default block
   length (primary).
4. Optionally report HLN–DM + Newey–West with `nw_lags = h − 1` (secondary).

**Wording (unchanged from M7, applied per horizon):** claim
“significantly better” / “significantly lower mean OOS QLIKE vs HAR” **only**
when the **primary bootstrap** rejects at α (default 0.05) **and** mean
ΔQLIKE &lt; 0. Point QLIKE rankings across horizons remain descriptive.

### 11.4  Cross-horizon summary

Study root (example):
`data/artifacts/multi-horizon-screen-{symbol}-{date}/`.

| Artifact | Role |
| -------- | ---- |
| `screen_meta.json` | Horizons; per-h embargo / NW / block length; models; α |
| `h{h}d/` | Per-horizon screen artifacts (`metrics.json`, `oos_losses.json`, `inference.json`, …) |
| `horizon_summary.json` | Rows keyed by `(horizon_days, model)`: QLIKE / MSE / MAE, mean ΔQLIKE, bootstrap CI / p, `significant_vs_baseline` |
| `report.html` | Study memo with **Skill by horizon** table + locked wording |

Do not mix unlabeled multi-horizon metrics in one table without a
`horizon_days` key.

### 11.5  Jump-robust features (stretch)

When the `jump` registry family is enabled (§2.6), the matrix includes
**daily jump-proportion** columns only (`jump_prop_*`), not high-frequency Barndorff–Nielsen–
Shephard estimators. Do not narrate them as tick-based jump variation in
the multi-horizon memo. Flagship ``vip screen-horizons --with jump``
(or ``vix,jump``) includes the family when matrices are rebuilt. Omit
``jump`` from ``--with`` for the default core (+ optional VIX-only) study.


## 12  Implied vs realized (Milestone 9)

Milestone 9 asks whether implied vol (via VIX) helps forecast forward RV
**as a feature**, **as a model**, or both, under the same walk-forward and
M7 inference contract as the rest of the platform. The Milestone 9 stretch
``rates`` family (``tnx_level`` / ``tnx_chg_1d``; §2.8) is an optional
screening covariate only — it is **not** part of the Implied vs realized
model claim.

### 12.1  Two separable questions

| Question | Mechanism | Where it appears |
| -------- | --------- | ---------------- |
| Does the **IV−RV gap** add predictive information? | Gap columns in the feature matrix; Ridge primary screen + permutation importance | Factor ranking; HTML “Top IV−RV gap features” when present |
| Can a **VIX-based forecast** compete with HAR? | Model `vix_as_forecast` in the horse-race vs baseline `har_rv_ols` | `metrics.json` / `inference.json`; HTML “VIX as competing forecast” |

Do **not** fold the full gap vector into `vix_as_forecast`. Do **not** claim
“implied beats realized” from a point QLIKE ranking alone.

### 12.2  Horse-race and inference

Catalog (when VIX predictors exist): `har_rv_ols`, `ridge`, `lasso`,
`vix_as_forecast`. If the matrix lacks `vix_vol_daily` and `vix_level`,
`vix_as_forecast` is omitted. Primary inference remains **moving block
bootstrap** of mean OOS ΔQLIKE vs `har_rv_ols` (M7/M8 defaults for h=5).
Optional HLN–DM stays secondary.

### 12.3  Report wording (unchanged from M7)

- “**Significantly** lower mean OOS QLIKE vs HAR” **only** when the **primary
  bootstrap** rejects at α (default 0.05) **and** mean ΔQLIKE &lt; 0
- Otherwise: lower / higher mean OOS QLIKE vs HAR (not significant at α)
- HTML section **Implied vs realized** restates the VIX proxy caveat, locked
  conversion, IV-model row, and optional top `vix_minus_rv_*` importance rows

### 12.4  CLI flagship extras

```text
--with vix          → vix_level, vix_chg_1d only
--with iv_rv        → implies VIX + IV−RV gap family
--with vix,iv_rv    → same as iv_rv for VIX load; gaps enabled
--with rates        → tnx_level, tnx_chg_1d (requires TNX ingest)
--with vix,iv_rv,rates → VIX + gaps + rates covariates
```

`vip screen` loads the existing processed matrix (no `--with`). Rebuild with
`vip features ... --with vix,iv_rv` / `--with rates` / `--with vix,iv_rv,rates`,
or the matching `vip run --with ...` form.

## 13  Caveats

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
  a default ``delta_cap = 1.0`` on factor screens (override with ``None`` for diagnostics) to mitigate this.
7. **Walk-forward assumptions.**  Expanding windows assume stationarity of the
  feature–target relationship.  Structural breaks (e.g. post-COVID liquidity
   regime) may violate this.
8. **Inference vs ranking.** Horse-race QLIKE orderings without bootstrap
   inference are descriptive, not findings.
9. **Overlap.** Overlapping RV labels require block bootstrap (and HAC lag
   h−1); i.i.d. day bootstrap understates dependence.
10. **Daily bipower ≠ tick bipower.** Jump-family columns are daily proxies for
    research screening only; they are not substitutes for high-frequency RV.
11. **VIX ≠ single-name IV.** Index VIX is a research proxy for ETF studies;
    it is not firm-level implied vol and not variance-swap replication (§2.7, §12).
12. **Feature vs model.** Gap importance and `vix_as_forecast` ΔQLIKE answer
    different questions; do not collapse them into one “IV beats RV” claim.
13. **Rates ≠ implied.** ``tnx_*`` covariates (§2.8) are yield level/change
    features for screening; they are not an IV proxy and do not enter
    ``vix_as_forecast``.