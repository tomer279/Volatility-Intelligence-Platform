# Milestone 10 Walkthrough — Parametric / Filter Baselines

## Objective

Add structurally different **parametric / filter baselines** for forward realized volatility — discrete OU-style mean reversion and (stretch) a simple recursive filter — and ask whether they beat **HAR-RV OLS** under the same walk-forward + M7 block-bootstrap contract.

This milestone should prove:

- The platform ships at least one **univariate parametric baseline** (`ou_rv`) that forecasts h-step RV from train-only dynamics, with a positive prediction floor and typed errors.
- That model enters the factor-screen **horse-race** beside `har_rv_ols` / Ridge / Lasso (and `vix_as_forecast` when VIX columns exist) and is scored with block-bootstrap ΔQLIKE.
- The HTML research memo answers: **does a mean-reverting / filter forecast beat HAR on OOS QLIKE?** with M7 wording discipline (“significantly better” only when bootstrap rejects at α and mean ΔQLIKE < 0).
- Optional stretch: a **recursive EWMA / filter** baseline that updates state from past RV only (not today’s frozen end-of-train `ewma`), and/or appendix diagnostics (rough-vol memory features, Granger/MI) that do **not** redefine the primary claim.

---

## Scope

### In scope

- Discrete OU / AR(1)-on-log-RV baseline in `vip.modeling` + model registry (`ou_rv`)
- Analytic (or explicitly iterated) **h-step** forecast mapped back to the platform target units
- Wire into `HORSE_RACE_MODELS` + existing M7 inference vs `har_rv_ols`
- Flagship single-horizon path (default **h=5**): `vip screen` / `vip run` — no new CLI feature-token required for core (model always eligible)
- Horizon awareness: when screens run at h ∈ {1, 5, 21}, the OU forecast horizon matches `target.horizon_days` / fold settings (reuse M8 injectables; do not rewrite multi-horizon orchestration)
- HTML section **“Parametric vs HAR”** (model row(s) + bootstrap gates + short caveats)
- Methodology + package docs; mark M10 in `plan.md` when exit met
- Unit tests (network-free): fit/predict, multi-step math on synthetic OU paths, leakage (no future target in state), horse-race wiring

### Stretch (same milestone if schedule allows; else later polish)

- Second baseline: recursive / rolling filter (e.g. `ewma_recursive` or `sv_filter`) — state updated with **past** RV only; distinct from frozen `EwmaModel`
- Optional rough-vol–inspired **feature** family (log-vol memory / fractional-ish lags) behind registry + leakage tests + CLI `--with` token
- Optional **diagnostics appendix** tooling (Granger causality / mutual information for selected feature → forward-RV pairs) — report as diagnostics, never as primary “beats HAR” claims
- Multi-horizon flagship narrative that includes `ou_rv` under `screen-horizons` (Skill by horizon already exists; just ensure the model is in the race)

### Out of scope

- Full continuous-time SV / Heston estimation, MLE toolchains, or options-pricing lab work
- GARCH/ARCH package zoo as the milestone center (a single simple GARCH-lite is not required; prefer OU + optional recursive EWMA)
- Intraday / high-frequency RV or true tick bipower
- Options-implied surfaces / single-name IV vendors
- Granger / MI as **primary** significance claims or replacements for M7 bootstrap
- Peer ETF / multi-horizon IV narrative leftovers from M9 (optional tiny polish only; not M10’s theme)
- Cross-sectional / portfolio-of-names models
- Live scheduling / production monitoring
- Hyperparameter search / Optuna
- Rewriting walk-forward, inference, or multi-horizon orchestration

---

## Acceptance Criteria

1. Model registry includes `ou_rv` implementing the same `fit` / `predict` surface as other horse-race models (`VolatilityModel`).
2. Locked research contract for `ou_rv` is implemented and unit-tested: state on **log target** (or documented alternate), train-only parameter fit, h-step mean forecast, exp-back + positive floor.
3. Leakage / integrity tests assert: no use of future target values in `fit` or `predict`; predict uses only information available at the forecast origin consistent with the walk-forward fold contract.
4. Flagship SPY screen (h=5) includes `ou_rv` in the horse-race; M7 block bootstrap vs `har_rv_ols` is reported for it.
5. HTML memo gains a **Parametric vs HAR** section: OU (and stretch filter if present) ΔQLIKE / bootstrap CI / p with locked wording; short discrete-time / physical-measure caveat.
6. Existing models remain intact: `historical_mean`, frozen `ewma`, `har_rv_ols`, `vix_as_forecast` (when VIX present), Ridge/Lasso/RF behavior unchanged aside from catalog membership.
7. Horizon-aware forecast: when `horizon_days` ≠ 5 (M8 path), `ou_rv` uses that h (constructor/config injection — not a hard-coded 5).
8. `docs/research_methodology.md` documents the discrete OU approximation, log-state choice, h-step mapping, and “must beat HAR under bootstrap to matter.”
9. Full pytest suite green; `plan.md` gains Milestone 10 DONE when criteria met.
10. **(Stretch)** Recursive filter baseline and/or rough-vol features and/or diagnostics appendix shipped with tests and docs — not blocking core DONE.

---

## Locked Research Defaults


| Setting                  | Value                                                                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Flagship symbol          | SPY                                                                                                                                                   |
| Batch symbols (optional) | SPY, QQQ                                                                                                                                              |
| Primary horizon          | **5** trading days (multi-horizon reuse via M8)                                                                                                       |
| Target                   | `target_rv_cc_5d` (or `target_rv_cc_{h}d` when multi-horizon)                                                                                         |
| Primary metric           | QLIKE (lower better)                                                                                                                                  |
| Secondary metrics        | MSE, MAE (descriptive)                                                                                                                                |
| Walk-forward             | expanding, `n_splits=5`, `embargo_size = horizon_days`                                                                                                |
| Inference baseline       | `har_rv_ols`                                                                                                                                          |
| Horse-race models (min)  | `har_rv_ols`, `ridge`, `lasso`, `vix_as_forecast` (when VIX cols), `ou_rv`                                                                            |
| Primary inference        | Moving block bootstrap of mean OOS ΔQLIKE (M7/M8)                                                                                                     |
| Secondary inference      | Optional HLN–DM + NW (`nw_lags = h − 1`)                                                                                                              |
| `alpha`                  | **0.05**                                                                                                                                              |
| Significance wording     | “Significantly better” **only** if bootstrap rejects at α **and** mean ΔQLIKE < 0                                                                     |
| Core model name          | `ou_rv`                                                                                                                                               |
| State variable           | **log** of training target (positive RV); see Research Contract                                                                                       |
| Forecast                 | Analytic h-step conditional mean on log-state → `exp` → floor                                                                                         |
| Prediction floor         | `1e-8` (same spirit as other baselines)                                                                                                               |
| Feature dependence       | **Univariate on target history** in `fit`; `predict` may ignore feature columns except for index alignment (like `EwmaModel` / `HistoricalMeanModel`) |
| CLI                      | No new core `--with` token; stretch rough-vol may add e.g. `rough`                                                                                    |
| Artifact root            | Existing screen layout under `data/artifacts/`                                                                                                        |


### Discrete OU contract (locked)

Work on x_t = \log y_t where y_t is the **training target** series (platform forward-RV labels available in-sample).

Discrete mean-reversion / AR(1):

```text
x_t = θ + φ (x_{t-1} − θ) + ε_t
```

Equivalently fit intercept + lag OLS of x_t on x_{t-1} and map to (\hatθ, \hatφ), with \hatφ \in (-1, 1) after a documented clip/guard if needed for stability.

**h-step conditional mean** from origin state x_T (last finite train log-target, or last available origin consistent with fold end):

```text
E[x_{T+h} | x_T] = θ + φ^h (x_T − θ)
ŷ_{T+h} = exp(E[x_{T+h} | x_T])   # then floor at prediction_floor
```

For walk-forward `predict` on a test block: each row’s forecast origin must use **only past** realized target values known by that origin under the evaluation contract. Preferred simple rule (lock and test one):

- **Frozen-origin style (MVP, preferred):** like current `EwmaModel`, fit (\hatθ, \hatφ) on the train fold; freeze the end-of-train log state x_T; emit the **same** h-step mean forecast for all rows in the test block (constant path forecast). Document that this matches the frozen-EWMA philosophy and avoids recursive use of test labels.
- **Recursive OOS style (stretch only):** update state through the test calendar using lagged **realized** y only when those y are strictly past the forecast origin and are not the label being predicted — easy to get wrong; do not make this the core path.

Rationale for MVP frozen-origin: parity with existing `EwmaModel`, fewer leakage footguns, still a distinct parametric story vs HAR’s three RV lags.

### Why these defaults


| Choice                             | Rationale                                                                        |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| Univariate OU on log-RV            | Classic vol mean-reversion story; few params; separable from HAR’s multi-lag OLS |
| Frozen end-of-train state for core | Matches platform’s frozen EWMA; safer under walk-forward                         |
| Always in horse-race               | No VIX dependency; cheap; always comparable to HAR                               |
| Keep HAR as baseline               | Design principle: alternatives must beat HAR on OOS QLIKE (+ bootstrap)          |
| Diagnostics as stretch             | Secondary to the forecast horse-race; avoid diluting M10’s claim                 |
| No GARCH zoo                       | Keeps milestone portfolio-ready and registry-thin                                |


---

## Target Folder Additions

```text
src/vip/
  modeling/
    baselines.py                 # EDIT: OuRvModel (or NEW parametric.py if file grows)
    registry.py                  # EDIT: register ou_rv (+ stretch filter)
    README.md
  application/
    screen_horse_race.py         # EDIT: add ou_rv to HORSE_RACE_MODELS
    screen_factors.py            # thin; reuse race helper
    # optional stretch only:
    # diagnostics_*.py           # Granger/MI helpers — not on critical path
  features/
    # stretch only:
    # rough_vol_features.py      # optional memory features
    # registry.py / pipeline.py / feature_extras CLI tokens
  reporting/
    experiment_summary.py        # EDIT: ParametricVsHar section inputs
    templates/
      factor_screen.html.j2      # EDIT: Parametric vs HAR section
    README.md
  cli/
    # no core token; stretch rough/diagnostics only if needed
    README.md

docs/
  research_methodology.md        # § Parametric / filter baselines (M10)
  milestones/
    milestone 10 walkthrough.md  # this document when committed

tests/unit/
  test_ou_rv_model.py            # fit/predict, h-step math, floors, errors
  test_ou_rv_leakage.py          # optional separate; or cases inside model tests
  test_ou_rv_screen_wiring.py    # horse-race includes ou_rv (tmp_path)
  # stretch:
  # test_ewma_recursive.py / test_rough_vol_features.py / test_diagnostics_*.py
```

Reuse — do **not** invent a parallel evaluation stack: extend the model registry and `HORSE_RACE_MODELS`, then call existing `run_horse_race_with_inference` / M7 summarization.

---

## Research Contract

### Competing forecast: `ou_rv`

**Role.** A simple parametric baseline: vol mean-reverts on a log scale; the h-step conditional mean is the forecast of forward RV — not a factor model and not a continuous-time estimator.

**Locked behavior:**

1. `fit(features, target)`:
  - Ignore feature columns for estimation (index may be used only for alignment).
  - Drop non-finite targets; require enough points for AR(1) (document minimum, e.g. ≥ 30 or raise `DataValidationError`).
  - Set x = \log(y) with y > 0 (values ≤ 0 / non-finite dropped or rejected — pick one and test).
  - Estimate (\hatθ, \hatφ) via OLS of x_t on x_{t-1} (intercept form), or equivalent closed form.
  - Store end-of-train state x_T = last finite training log-target.
  - Store `horizon_days` from constructor (default 5; injectable for M8).
2. `predict(features)`:
  - Require fitted params + state.
  - Compute \hatμ = \hatθ + \hatφ^h (x_T - \hatθ), \hat y = \max(\exp(\hatμ), \text{floor}).
  - Return a Series aligned to `features.index` (constant forecast under MVP frozen-origin rule).
3. Raise typed `DataValidationError` on empty train, non-positive series, unstable unusable \hatφ if you refuse to clip, or unfitted predict.
4. Register as `ou_rv` in `create_default_model_registry()`.
5. Add to `HORSE_RACE_MODELS` unconditionally (unlike `vix_as_forecast`, which is skipped without VIX columns).

**Non-goals for this model:** using HAR lag columns inside OU; tree methods; hyperparameter search; estimating a full diffusion with MLE libraries.

### Stretch: `ewma_recursive` (or `sv_filter`)

If schedule allows:

1. Fit decay (or noise ratio) on train targets only.
2. Predict with a state that is **not** merely the single frozen EWMA level copied to all test rows **unless** you explicitly document parity — prefer a clearly named model that either (a) still freezes end-of-train level but uses OU-incompatible update math you want to compare, or (b) implements a documented recursive rule without reading the label at the forecast origin.
3. Must remain distinct from existing registry name `ewma`.
4. Same horse-race + bootstrap wiring; same HTML section rows.

### Screen + inference

For the flagship h=5 run:

1. Build/load feature matrix as today (VIX / iv_rv / jump / rates optional extras unchanged).
2. Run horse-race including `ou_rv`.
3. Persist OOS losses; run block bootstrap ΔQLIKE vs `har_rv_ols` for each challenger (including OU).
4. Wording: never claim “OU beats HAR” from point QLIKE alone.
5. Design principle reminder in report/methodology: parametric novelty without OOS skill is not a finding.

### CLI


| Token                                       | Effect                                  |
| ------------------------------------------- | --------------------------------------- |
| (none new for core)                         | `ou_rv` always in race                  |
| `rough`                                     | Stretch only — rough-vol feature family |
| existing `vix` / `jump` / `iv_rv` / `rates` | Unchanged                               |


### Artifacts / report


| Artifact / section          | Content                                                        |
| --------------------------- | -------------------------------------------------------------- |
| `metrics.json` / horse-race | Row for `ou_rv` (+ stretch filter)                             |
| `inference.json`            | Bootstrap CI/p for `ou_rv` vs HAR                              |
| HTML **Parametric vs HAR**  | Discrete-OU caveat; ΔQLIKE + significance; optional filter row |
| Methodology                 | Log-state; h-step formula; frozen-origin MVP; physical measure |


---

## Design Rules

1. **Reuse** model registry, `screen_horse_race`, walk-forward, M7 inference, HTML template patterns from M9’s Implied section — M10 is modeling + reporting, not a new pipeline.
2. **Univariate baseline ≠ feature family** — keep OU as a model; rough-vol memory (if any) as optional features.
3. **Do not break** frozen `ewma` or `vix_as_forecast` resolve logic.
4. Bootstrap remains **primary**; HLN–DM secondary; wording unchanged from M7.
5. CLI thin; math in `modeling` (and stretch `features`); orchestration stays in `application`.
6. NumPy docstrings; module/class docs; ≤5 params (bundle horizon/floor in constructor or small frozen config); ≥2 public methods per class; no broad `except`; typed domain errors; no unused typing imports.
7. Avoid magic numbers in array indexing; name constants (`DEFAULT_OU_MIN_OBS`, `DEFAULT_PREDICTION_FLOOR`, etc.).
8. Do not mutate caller frames/series inside helpers.
9. Horizon injection must not hard-code 5 inside `predict` math.
10. Stretch diagnostics must not weaken horse-race wording or replace QLIKE + bootstrap.

---

## Step-by-Step Build Plan

### Step 1 — `OuRvModel` + unit tests

Implement in `baselines.py` (or `parametric.py` if cleaner):

- Constructor: `horizon_days`, `prediction_floor` (and optional `min_obs`)
- `fit` / `predict` per locked contract
- Helpers for log transform, AR(1)/OU param map, h-step mean

Unit tests:

- Synthetic AR(1) on log-vol → fitted \hatφ near truth
- Known (θ, φ, x_T, h) → exact expected prediction (tolerance)
- Floor applied; errors on empty / unfitted / non-positive

**Checkpoint:** `test_ou_rv_model.py` green.

### Step 2 — Integrity / leakage tests

Assert:

- Permuting **future** target values after the train end does not change fitted params/state when `fit` sees only train.
- `predict` does not read `target` at all.
- Under frozen-origin MVP, predictions are constant across the test index.

**Checkpoint:** leakage/integrity cases green.

### Step 3 — Registry + horse-race wiring

- Register `ou_rv` in `create_default_model_registry()`.
- Append `"ou_rv"` to `HORSE_RACE_MODELS`.
- Ensure `resolve_horse_race_models` still drops only `vix_as_forecast` when VIX absent — never drop `ou_rv`.
- Horizon: factory or screen path must pass `horizon_days` into the model when h ≠ 5 (bundle via existing settings helpers if needed; keep ≤5 params).

**Checkpoint:** `test_ou_rv_screen_wiring.py` (or extend existing wiring tests) shows `ou_rv` in summary/inference.

### Step 4 — HTML + wording

Add **Parametric vs HAR** section (mirror Implied vs realized pattern):

- Short caveat (discrete OU; log-state; not continuous SV; frozen-origin MVP).
- Table/rows for `ou_rv` vs HAR (ΔQLIKE, CI, p, significant flag).
- Stretch: additional filter row when present.

**Checkpoint:** template/report test asserts section presence and wording helper reuse.

### Step 5 — Methodology + package docs + `plan.md`

Update:

- `docs/research_methodology.md` — new § (e.g. §13 Parametric / filter baselines)
- `src/vip/modeling/README.md`, application/reporting READMEs as needed
- `plan.md` — full Milestone 10 section; status → DONE only at exit
- Commit this walkthrough under `docs/milestones/milestone 10 walkthrough.md` when implementing

**Checkpoint:** docs match locked names/formulas.

### Step 6 — Flagship run + exit

```powershell
$env:PYTHONPATH = "src"
py -m pip install -e ".[dev]"
vip ingest --symbol SPY
vip features --symbol SPY --horizon 5
vip screen --symbol SPY
# inspect report.html → Parametric vs HAR
py -m pytest -q
```

Optional:

```powershell
vip screen-horizons --symbol SPY
vip run --symbol SPY --with vix,iv_rv
```

**Checkpoint:** acceptance criteria 1–9 met; `plan.md` M10 DONE.

### Step 7 — Stretch (optional)

Recursive filter model and/or rough-vol features and/or Granger/MI diagnostics appendix — each with tests; none block core DONE.

---

## Suggested Command Sequence

```powershell
$env:PYTHONPATH = "src"
py -m pip install -e ".[dev]"

py -m pytest tests/unit/test_ou_rv_model.py -q
py -m pytest tests/unit/test_ou_rv_screen_wiring.py -q

vip ingest --symbol SPY
vip features --symbol SPY --horizon 5
vip screen --symbol SPY
vip run --symbol SPY

# inspect data/artifacts/**/report.html (Parametric vs HAR)
py -m pytest -q
```

Optional multi-horizon / extras:

```powershell
vip screen-horizons --symbol SPY --with vix,iv_rv
```

---

## Common Pitfalls

- Fitting OU on **raw** RV without log transform, then exponentiating anyway — inconsistent state and unstable φ.
- Using **forward** target labels inside recursive predict (leakage).
- Emitting a 1-step forecast when the evaluation target is 5-day (or 21-day) RV — horizon must match.
- Registering `ou_rv` but forgetting `HORSE_RACE_MODELS`, so the report never shows inference.
- Accidentally skipping `ou_rv` in `resolve_horse_race_models` with the VIX gate.
- Claiming “OU beats HAR” from a QLIKE ranking without bootstrap gates.
- Turning M10 into a GARCH/SV research lab or adding Optuna.
- Letting Granger/MI diagnostics become the headline claim.
- Rewriting multi-horizon orchestration instead of injecting `horizon_days` into the model.
- Mutating the training series in place or using magic index literals without named constants.

---

## Decisions Locked for This Walkthrough

1. **M10 theme = parametric / filter baselines** vs HAR on the existing daily ETF spine.
2. Core model name: `ou_rv` — discrete OU / AR(1) on **log** target; analytic h-step mean; `exp` + floor.
3. Core predict style: **frozen end-of-train state** (MVP); recursive OOS updates are stretch-only.
4. `ou_rv` is **always** in the horse-race (no feature-column gate).
5. Inference baseline remains `har_rv_ols`; M7 bootstrap primary; wording unchanged.
6. Primary flagship horizon remains **5**; M8 multi-horizon reuse via injectable `horizon_days`.
7. No new core CLI `--with` token; stretch may add `rough` / diagnostics.
8. Stretch: recursive filter and/or rough-vol features and/or Granger/MI appendix only.
9. Out of scope: options surfaces, HF RV, continuous SV pricing lab, cross-section, scheduling, Optuna, orchestration rewrites.

---

## Milestone 10 Exit Checklist

- [x] `OuRvModel` (`ou_rv`) + unit tests (fit/predict/h-step/floor/errors)
- [x] Integrity / leakage tests (train-only fit; predict ignores target; frozen-origin behavior)
- [x] Registry + `HORSE_RACE_MODELS` includes `ou_rv` unconditionally
- [x] Horizon injection works for h ∈ {1, 5, 21} on the model path
- [x] Horse-race + M7 inference artifacts include `ou_rv`
- [x] HTML “Parametric vs HAR” section + locked wording
- [x] Methodology + READMEs + `plan.md` Milestone 10 section
- [x] Flagship SPY run inspected; full `pytest -q` green; plan status DONE
- [x] (Stretch) recursive filter baseline + tests + HTML row
- [ ] (Stretch) rough-vol feature family + leakage tests + CLI token
- [ ] (Stretch) Granger/MI diagnostics appendix (non-claim)

**Acceptance map:** checklist rows 1–3 ↔ criteria 1–3. Row 4 ↔ criterion 7. Rows 5–6 ↔ criteria 4–5. Row 7 ↔ criteria 6 & 8. Row 8 ↔ criterion 9 exit gate. Remaining rows ↔ criterion 10 (stretch; not blocking core DONE).

---

## What Comes Next (post-M10)

Ordered backlog after parametric baselines:

1. Optional diagnostics / rough-vol memory if not finished as M10 stretch
2. Broader cross-asset covariates (peer ETF narrative deferred from M9)
3. Event studies once a calendar source exists; MC scenario bands as evaluation appendix

Still deferred: intraday HF RV, options surfaces/Greeks, sentiment, cross-sectional models, live scheduling.