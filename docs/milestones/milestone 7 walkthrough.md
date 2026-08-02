# Milestone 7 Walkthrough — Statistical Inference on OOS Gaps

## Objective

Turn descriptive horse-race rankings into portfolio-defensible claims: attach uncertainty to mean OOS ΔQLIKE vs the HAR baseline, accounting for overlapping multi-day RV labels.

This milestone should prove:

- Per-observation (per-row) OOS QLIKE losses are persisted alongside fold aggregates — inference is not run on fold means alone.
- **Primary** inference is a **block bootstrap** of mean OOS ΔQLIKE vs `har_rv_ols` (default block length 15 trading days), reporting mean gap, bootstrap CI, and bootstrap p-value.
- **Secondary** (optional but recommended): Diebold–Mariano with Newey–West HAC **and** Harvey–Leybourne–Newbold (HLN) finite-sample correction — never ship uncorrected DM as the sole claim.
- Report wording is locked: “significantly better” only when the **primary** (bootstrap) test rejects at α; otherwise “lower mean OOS QLIKE”.
- Results land in `metrics.json` / comparison tables / HTML memo; methodology documents overlap, effective sample size, bootstrap, NW lag, and optional HLN–DM.

---



## Scope



### In scope

- `vip.evaluation.metrics` — per-observation QLIKE (elementwise loss series; mean matches existing `qlike`)
- `vip.evaluation.inference` — loss differentials, block bootstrap, optional HLN–DM + Newey–West
- Extend walk-forward path to emit / reuse per-row OOS losses (prefer building on `collect_walk_forward_predictions`)
- Extend `summarize_walk_forward` / comparison helpers (or a thin `summarize_with_inference`) so horse-race tables carry mean ΔQLIKE + bootstrap CI / p-value
- Wire into `screen_factors` (and `vip evaluate` / `run_baseline_experiment` where a comparison table already exists): persist `oos_losses.json` (or equivalent), enrich `metrics.json`, update HTML + caveats
- Optional sensitivity: non-overlapping every-`horizon`-day subsample (footnote in report / methodology)
- Unit tests on synthetic loss differentials (known null / known alternative); network-free
- Document overlap, effective N, block bootstrap, NW lag = `horizon_days - 1`, optional HLN–DM in `docs/research_methodology.md`
- Update `plan.md` M7 DONE when exit criteria met



### Out of scope

- Options-implied surfaces
- Intraday / high-frequency RV
- Cross-sectional / portfolio-of-names models
- Live scheduling / production monitoring
- Hyperparameter search / Optuna
- Rewriting feature engineering
- Treating fold-mean-only QLIKE as sufficient for statistical claims
- Shipping plain DM (without HLN) as the primary or sole significance claim

---



## Acceptance Criteria

1. Per-row OOS QLIKE losses are available for each horse-race model on the same walk-forward test rows (indexed by session date); persisted under the experiment artifact dir (e.g. `oos_losses.json`).
2. For each horse-race model vs baseline `har_rv_ols`, the platform computes mean ΔQLIKE = mean(L_model − L_baseline) on aligned OOS rows (negative = model better under QLIKE).
3. **Block bootstrap is required for exit:** default block length **15** (configurable in **10–20**), seeded; reports mean ΔQLIKE, bootstrap CI at configured α (default 0.05), and bootstrap two-sided p-value for H0: mean ΔQLIKE = 0.
4. Newey–West lag for any HAC / DM path is locked to `horizon_days - 1` (**4** for `target_rv_cc_5d`).
5. Optional HLN–DM path: if enabled, report DM statistic, HLN-corrected statistic / p-value alongside bootstrap — **do not** claim significance from uncorrected DM alone; primary wording still follows bootstrap.
6. `metrics.json` (and CLI comparison table) includes inference columns for each non-baseline model: `mean_delta_qlike`, `bootstrap_ci_low`, `bootstrap_ci_high`, `bootstrap_pvalue` (plus optional HLN–DM fields).
7. HTML research memo shows the inference-enriched horse-race and uses locked wording (“significantly better” iff primary bootstrap rejects at α; else “lower / higher mean OOS QLIKE”).
8. Unit tests: synthetic differentials under a known null do not systematically reject at α; under a known alternative, bootstrap rejects with high probability; block bootstrap ≠ i.i.d. resample of single days; NW lag helper returns `horizon_days - 1`.
9. `docs/research_methodology.md` documents label overlap, why embargo ≠ inference, block bootstrap primary, NW lag, optional HLN–DM, and wording rules.
10. Full pytest suite green; no network in unit tests; `plan.md` M7 marked DONE.

---



## Locked Research Defaults


| Setting                  | Value                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Flagship symbol          | SPY (batch SPY/QQQ unchanged)                                                                                    |
| Target                   | `target_rv_cc_5d` (`horizon_days=5`)                                                                             |
| Primary metric           | QLIKE (lower better)                                                                                             |
| Secondary metrics        | MSE, MAE (descriptive; inference on QLIKE differentials)                                                         |
| Walk-forward             | expanding, `n_splits=5`, `embargo=5`                                                                             |
| Inference baseline       | `har_rv_ols`                                                                                                     |
| Horse-race models (min)  | existing screen set (`har_rv_ols`, `ridge`, `lasso`, plus RF when registered in the race)                        |
| Loss differential        | `d_t = L_t(model) − L_t(baseline)` on OOS rows; mean ΔQLIKE = mean(`d_t`)                                        |
| **Primary inference**    | Moving **block bootstrap** of mean(`d_t`)                                                                        |
| `bootstrap_block_length` | **15** trading days                                                                                              |
| `bootstrap_block_range`  | **10–20** (allowed config; default 15)                                                                           |
| `bootstrap_n_resamples`  | ≥ 999 (recommend 1999 for reports; 999 OK for unit tests)                                                        |
| `bootstrap_random_seed`  | `0`                                                                                                              |
| `alpha`                  | **0.05**                                                                                                         |
| Bootstrap CI             | Percentile (or BCa if cheap); two-sided                                                                          |
| Bootstrap p-value        | Two-sided vs H0: E[`d_t`] = 0                                                                                    |
| `nw_lags`                | `horizon_days - 1` **→ 4** for default 5-day target                                                              |
| **Secondary inference**  | Optional HLN–DM + Newey–West (`nw_lags=4`); never sole claim                                                     |
| Significance wording     | “Significantly better” **only** if bootstrap rejects at α                                                        |
| Optional sensitivity     | Non-overlapping subsample: keep every `horizon_days`-th OOS row (footnote)                                       |
| Artifact root            | Reuse existing `data/artifacts/factor-screen-{symbol}-{date}/` (and baseline experiment dirs for `vip evaluate`) |




### Why these defaults


| Choice                          | Rationale                                                                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Block bootstrap primary         | ~5 walk-forward folds make fold-level DM anti-conservative; QLIKE differentials are asymmetric — bootstrap handles that better than a normal DM p-value |
| Block length 10–20 / default 15 | Captures short-run serial dependence from overlapping 5-day labels without devouring the OOS sample                                                     |
| NW lag = horizon − 1            | Standard HAC lag for overlapping h-step forecast errors                                                                                                 |
| HLN on DM                       | Small effective samples; uncorrected DM over-rejects — HLN is required if DM is reported                                                                |
| Baseline = HAR OLS              | Locked research reference since M3/M4 horse-races                                                                                                       |


---



## Target Folder Additions

```text
src/vip/
  evaluation/
    metrics.py              # add qlike_losses (elementwise) — keep scalar qlike
    inference.py            # NEW: differentials, block bootstrap, optional HLN–DM
    walk_forward.py         # emit/reuse per-row OOS losses (or thin helper beside collect_*)
    comparison.py           # enrich summary with inference columns
    __init__.py             # export inference APIs
    README.md               # document inference contract
  application/
    screen_factors.py       # run inference after horse-race; persist artifacts
    run_baseline_experiment.py  # optional: same inference on evaluate path
  reporting/
    experiment_summary.py   # inference rows + wording + caveats
    templates/
      factor_screen.html.j2 # horse-race columns + significance note
  cli/commands/
    screen.py               # print ΔQLIKE + bootstrap CI/p (optional flags)
    evaluate.py             # optional parity

docs/
  research_methodology.md   # M7 section: overlap, bootstrap, NW, HLN–DM, wording
  milestones/
    milestone 7 walkthrough.md

tests/unit/
  test_qlike_losses.py          # elementwise mean == scalar qlike
  test_inference_bootstrap.py   # null / alternative / block vs iid
  test_inference_dm_hln.py      # NW lag = 4; HLN correction applied
  test_inference_wiring.py      # comparison / screen artifact shape (tmp_path)
```

Reuse — do **not** invent a parallel pipeline: same folds, same `create_default_model_registry` models, same artifact store, same HTML memo path.

---



## Research Contract



### Per-observation QLIKE

Extend `metrics.py` with an elementwise helper, e.g.:

```python
def qlike_losses(
    y_true: pd.Series,
    y_pred: pd.Series,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.Series:
    """Per-row QLIKE; mean equals qlike(...)."""
```

Formula per row (same as existing scalar):

```text
L_t = log(ŷ_t²) + y_t² / ŷ_t²
```

with `ŷ_t = max(ŷ_t, ε)`. Assert in tests: `qlike_losses(...).mean() ≈ qlike(...)`.

### OOS loss panel

Build on `collect_walk_forward_predictions` (already returns `model`, `fold_id`, `y_true`, `y_pred`, date index):

1. For each model’s OOS rows, compute `qlike_loss` via `qlike_losses`.
2. Persist long-form panel, e.g. columns:
  `date`, `model`, `fold_id`, `y_true`, `y_pred`, `qlike_loss`.
3. Prefer one shared helper (e.g. `collect_walk_forward_oos_losses(...)`) so regime scoring and inference share the same predictions — avoid a second fit loop if practical (time-box: a second pass is OK for M7 exit if documented).



### Loss differential vs baseline

For challenger model `m` and baseline `b = har_rv_ols`:

1. Inner-join OOS rows on date (and optionally `fold_id`) so `d_t` is paired.
2. `d_t = L_t(m) − L_t(b)`.
3. `mean_delta_qlike = mean(d_t)`. Lower (more negative) means `m` has lower mean OOS QLIKE than baseline.

**Do not** difference fold-mean QLIKE values and call that inference — use the per-row series.

### Primary: block bootstrap

Config dataclass (keep public call sites ≤5 params), e.g. `BootstrapInferenceOptions`:


| Field          | Default                      |
| -------------- | ---------------------------- |
| `block_length` | 15                           |
| `n_resamples`  | 1999 (tests may use 399–999) |
| `alpha`        | 0.05                         |
| `random_seed`  | 0                            |


Procedure (moving / circular block bootstrap of the `d_t` series in time order):

1. Let `T = len(d)`, `ℓ = block_length`, `μ̂ = mean(d)`.
2. For `b = 1..B`: draw contiguous blocks of length `ℓ` with replacement until length ≥ T; truncate to T; compute `μ*_b = mean(d*)`.
3. Percentile CI: empirical α/2 and 1−α/2 quantiles of `μ*_b`.
4. Two-sided bootstrap p-value for H0: E[d] = 0 — use a standard recentering / |μ̂| comparison against the bootstrap distribution of mean(d* − μ̂) or equivalent documented formula; unit-test against synthetic null/alternative.
5. Reject H0 at level α if p ≤ α (primary gate for “significantly better” when μ̂ < 0).

Validate `block_length` ∈ [10, 20] at config time (or warn and clamp — prefer hard validate for research defaults).

### Secondary: HLN–DM + Newey–West

Optional path (flag or always-compute-if-cheap):

1. Diebold–Mariano on `d_t` with HAC variance, Newey–West lags = `nw_lags = horizon_days - 1` (**4**).
2. Apply Harvey–Leybourne–Newbold finite-sample correction to the DM statistic before mapping to a Student-t (or documented) p-value.
3. Persist `dm_stat`, `hln_stat`, `hln_pvalue`, `nw_lags` as **secondary** columns.
4. Report text must not treat HLN–DM rejection alone as the primary “significantly better” claim when bootstrap does not reject (or document both and still gate marketing language on bootstrap).

Citations to name in methodology (no need to vendor papers in-repo):

- Diebold & Mariano (1995) — comparing predictive accuracy  
- Newey & West (1987) — HAC  
- Harvey, Leybourne & Newbold (1997) — small-sample DM correction  
- Overlapping multi-step forecasts → HAC lag ≈ h−1



### Optional sensitivity (footnote)

Subsample OOS dates to non-overlapping horizon spacing (every 5th trading day for h=5). Recompute mean ΔQLIKE ± bootstrap on the thinner series; report as footnote / `inference_sensitivity.json` — not a second primary claim.

### Artifacts written


| Artifact           | Content                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| `oos_losses.json`  | Per-row OOS losses (and optionally y_true/y_pred)                                              |
| `metrics.json`     | Horse-race means **plus** inference columns vs baseline                                        |
| `inference.json`   | Optional dedicated table: one row per challenger (mean Δ, CI, p, optional HLN–DM)              |
| `folds.json`       | Unchanged fold aggregates                                                                      |
| `report.html`      | Enriched horse-race + wording + methodology bullets                                            |
| `screen_meta.json` | Record `baseline_model`, `nw_lags`, `bootstrap_block_length`, `alpha`, `bootstrap_n_resamples` |




### Comparison / HTML schema (minimum columns)

For each non-baseline model row:


| Column                                   | Role                                                    |
| ---------------------------------------- | ------------------------------------------------------- |
| `model`                                  | Challenger name                                         |
| `qlike` / `mse` / `mae`                  | Existing mean OOS metrics                               |
| `mean_delta_qlike`                       | vs `har_rv_ols`                                         |
| `bootstrap_ci_low` / `bootstrap_ci_high` | Primary CI                                              |
| `bootstrap_pvalue`                       | Primary p                                               |
| `significant_vs_baseline`                | bool from bootstrap @ α **and** mean_delta < 0 (better) |
| `hln_pvalue` (optional)                  | Secondary                                               |


Baseline row: metrics only; Δ / p blank or zero by definition.

Wording helper (reporting): if `significant_vs_baseline` → “significantly lower mean OOS QLIKE vs HAR (bootstrap)”; elif mean_delta < 0 → “lower mean OOS QLIKE vs HAR (not significant at α)”; else analogous for higher.

---



## Design Rules

1. **Reuse** `run_walk_forward` / `generate_expanding_folds` / `collect_walk_forward_predictions` — no second CV engine.
2. **Leakage:** inference uses OOS test rows only; never refit inside bootstrap; embargo remains for train/test separation and is **not** a substitute for inference.
3. **Overlap:** overlapping `target_rv_cc_5d` labels induce serial dependence in `d_t` — use **block** bootstrap, not i.i.d. day bootstrap.
4. **NW lag** = `horizon_days - 1` (4). Do not default to rule-of-thumb `T^(1/3)` for this research path.
5. **Bootstrap is primary.** HLN–DM is secondary. Do not ship DM without HLN as the sole significance claim.
6. **Wording:** “significantly better” only when primary bootstrap rejects at α (and the gap favors the challenger). Mean gaps without rejection → “lower mean OOS QLIKE”.
7. Fold-mean QLIKE rankings remain descriptive; do not treat them as inferential findings.
8. CLI thin; orchestration in application; math in `evaluation.inference` / `metrics`.
9. NumPy docstrings; module/class docs; ≤5 params (bundle options in frozen dataclasses); no broad `except`; typed domain errors.
10. Do not mutate caller Series/DataFrames inside bootstrap helpers.

---



## Step-by-Step Build Plan



### Step 1 — Elementwise QLIKE (`metrics.py`)

Add `qlike_losses(...) -> pd.Series`.  
Unit test: empty/misaligned raises; mean matches `qlike` within float tolerance; epsilon floor applied.  
**Checkpoint:** `test_qlike_losses.py` green.

### Step 2 — OOS loss panel from walk-forward

Add `collect_walk_forward_oos_losses(...)` (or extend prediction collector) returning long-form losses for all horse-race models on identical folds/embargo.  
Prefer sharing fits with `collect_walk_forward_predictions` if easy; otherwise document a second pass.  
**Checkpoint:** synthetic panel has one row per test date × model; losses finite.

### Step 3 — Inference module skeleton (`inference.py`)

Implement:

- `nw_lags_for_horizon(horizon_days: int) -> int` → `horizon_days - 1`
- `loss_differential(challenger_losses, baseline_losses) -> pd.Series`
- `BootstrapInferenceOptions` + `block_bootstrap_mean(d, options) -> BootstrapResult`
- Optional: `hln_diebold_mariano(d, nw_lags) -> DMResult`

Keep public functions ≤5 params via options dataclasses.  
**Checkpoint:** imports + NW lag unit test.

### Step 4 — Block bootstrap correctness tests

Synthetic series:

1. **Null:** `d_t` mean-zero with mild AR dependence — empirical rejection rate of bootstrap test near α (Monte Carlo, modest B/reps for CI speed).
2. **Alternative:** large negative mean — high power / rejects.
3. **Block vs i.i.d.:** construct overlapping-style dependence where i.i.d. bootstrap understates variance; assert block CI is wider or p-values less anti-conservative than naive iid (document the exact assertion).

**Checkpoint:** `test_inference_bootstrap.py` green.

### Step 5 — Optional HLN–DM path

Implement DM + NW(4) + HLN.  
Tests: `nw_lags_for_horizon(5) == 4`; HLN statistic differs from raw DM on small T; p-value in (0, 1).  
**Checkpoint:** `test_inference_dm_hln.py` green; path skippable via config flag if desired.

### Step 6 — Comparison enrichment (`comparison.py`)

`summarize_with_inference(fold_metrics_or_oos_losses, baseline="har_rv_ols", bootstrap_options=..., include_hln_dm=True)` → summary DataFrame with locked columns.  
Baseline row without p-values; challengers filled.  
**Checkpoint:** unit test on tiny aligned loss panel.

### Step 7 — Wire `screen_factors` (+ optional `evaluate`)

After horse-race:

1. Collect OOS losses.
2. Run bootstrap (and optional HLN–DM) for each challenger vs `har_rv_ols`.
3. Persist `oos_losses.json`, enrich `metrics.json`, optional `inference.json`, update `screen_meta` with inference defaults.
4. Pass inference rows into `experiment_summary` / HTML template.
5. Add caveats: rankings without inference are descriptive; overlap; embargo ≠ significance; bootstrap primary.

CLI: print mean ΔQLIKE + CI + p for each challenger.  
**Checkpoint:** `vip screen --symbol SPY` writes enriched artifacts; `test_inference_wiring.py` green.

### Step 8 — HTML + wording

Extend `factor_screen.html.j2` horse-race table with ΔQLIKE, bootstrap CI, p-value, and a short note on significance language.  
Locked methodology list: add NW lag, block length, α, baseline.  
**Checkpoint:** `test_html_report.py` asserts new headings/columns present.

### Step 9 — Optional non-overlapping sensitivity

Helper to thin the OOS index every `horizon_days`; re-run bootstrap; persist footnote fields. Time-box if schedule slips — mark stretch but keep API hook.  
**Checkpoint:** unit test on synthetic dates.

### Step 10 — Methodology + plan status

Update `docs/research_methodology.md` (new section: Statistical inference on OOS gaps).  
Update `evaluation/README.md` exports.  
Mark M7 DONE in `plan.md` when acceptance criteria met.  
**Checkpoint:** docs consistent with code defaults (bootstrap primary — not “optional robustness only”).

---



## Suggested Command Sequence

```powershell
$env:PYTHONPATH = "src"
py -m pip install -e ".[dev]"
py -m pytest tests/unit/test_qlike_losses.py -q
py -m pytest tests/unit/test_inference_bootstrap.py tests/unit/test_inference_dm_hln.py -q
py -m pytest tests/unit/test_inference_wiring.py -q
vip features --symbol SPY --with-vix
vip screen --symbol SPY # inspect data/artifacts/factor-screen-spy-*/metrics.json and report.html
vip run --symbol SPY --with-vix --skip-ingest   # optional: confirm composite path still green
py -m pytest -q
```

---



## Common Pitfalls

- Running DM on **five fold-mean** QLIKE values and treating the p-value as valid — anti-conservative; use per-row `d_t` and treat bootstrap as primary.
- Shipping **DM without HLN** as the headline significance claim.
- Setting **NW lags ≠ 4** for the 5-day target (or “auto” lag that ignores horizon overlap).
- Claiming “significantly better” from a **mean QLIKE gap alone** (no CI / no primary test).
- Using **i.i.d. bootstrap** (resample single days) despite overlapping RV labels — understates dependence; use **block** bootstrap with length in 10–20.
- Bootstrapping **fold aggregates** instead of the OOS loss series.
- Differencing models on **misaligned** calendars (inner-join on dates).
- Treating **embargo** as proof the gap is real — it only blocks train/test leakage.
- Re-fitting models inside bootstrap replicates.
- Letting HTML/CLI still say “best model” without the significance wording rules.
- Building a **parallel** evaluation pipeline instead of extending horse-race / artifacts / HTML.

---



## Decisions Locked for This Walkthrough

1. **Primary inference = block bootstrap** of mean OOS ΔQLIKE vs `har_rv_ols` (block length default **15**, range **10–20**).
2. **NW lags =** `horizon_days - 1` **= 4** for the default 5-day target (any HAC / DM path).
3. **HLN–DM is secondary** (optional but recommended if cheap); never sole claim; never DM without HLN.
4. Persist **per-row OOS losses**; do not infer from fold means alone.
5. “Significantly better” **only** when the **primary bootstrap** rejects at α (default **0.05**) and the gap favors the challenger; otherwise “lower mean OOS QLIKE”.
6. Optional non-overlapping every-horizon subsample is a **footnote**, not a second primary test.
7. Reuse existing screen / evaluate / HTML artifact paths — no parallel research stack.
8. Out of scope: options, intraday RV, cross-section, scheduling, hyperparameter search, feature-engineering rewrite.

---



## Milestone 7 Exit Checklist

- [x] `qlike_losses` + tests (mean matches scalar `qlike`)
- [x] Per-row OOS loss panel persisted (`oos_losses.json`)
- [x] Block bootstrap (default ℓ=15) → mean ΔQLIKE, CI, p-value vs `har_rv_ols`
- [x] `nw_lags = 4` enforced for HAC / DM path
- [x] Optional HLN–DM secondary columns (no uncorrected-DM-only claims)
- [x] `metrics.json` / comparison table / HTML memo wired with locked wording
- [x] Unit tests: synthetic null / alternative; block ≠ naive iid; NW helper
- [x] `docs/research_methodology.md` updated (overlap, bootstrap primary, NW, HLN–DM)
- [x] Full pytest green; `plan.md` M7 DONE
- [x] (Stretch) non-overlapping horizon subsample footnote

---



## What Comes Next (post-M7)

Post-MVP research extensions remain explicitly out of scope until prioritized: intraday / high-frequency RV, options-implied surfaces, portfolio-of-names / cross-sectional models, live scheduling, and production monitoring. M7 closes the “is this OOS gap real?” gap for the existing daily ETF horse-race.