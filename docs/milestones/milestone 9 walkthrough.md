# Milestone 9 Walkthrough — IV−RV Gap & Implied-as-Forecast

## Objective

Treat implied volatility seriously on the existing daily ETF spine: move beyond
`vix_level` / `vix_chg_1d` as optional covariates and ask whether **the IV−RV
gap** adds predictive information, and whether a **VIX-based forecast** can
compete with HAR under the same walk-forward + M7 inference contract.

This milestone should prove:

- The platform builds an **IV−RV feature family** (VIX as IV proxy) with
documented unit alignment, as-of joins, and leakage tests.
- At least one **implied-as-forecast** model enters the horse-race beside
`har_rv_ols` / Ridge / Lasso and is scored with block-bootstrap ΔQLIKE.
- The HTML research memo answers: **does implied vol (or the gap) help forecast
forward RV — as a feature, as a model, or both?** with M7 wording discipline.
- Optional stretch: one small **rates / peer** cross-asset family behind the
same as-of join pattern (still yfinance; no new vendor).

---



## Scope



### In scope

- IV−RV gap feature family (registry + builders) using existing VIX OHLCV and
trailing HAR RV columns (`rv_cc_1d` / `rv_cc_5d` / `rv_cc_21d`)
- Documented **unit conversion** between VIX prints and platform RV (non-
annualized close-to-close trailing vol)
- Leakage tests: gap features use information set ≤ *t* only (VIX as-of
backward; trailing RV ending at *t*)
- Competing forecast model in `vip.modeling` + model registry, e.g.
`vix_as_forecast` (scaled VIX → forward RV)
- Wire into factor screen horse-race + M7 inference vs `har_rv_ols`
- Flagship single-horizon path (default **h=5**): `vip features` / `vip screen`
/ `vip run` with VIX + IV−RV extras
- Optional reuse of `vip screen-horizons` when the single-horizon path is
stable (same models/features per horizon; no new inference theory)
- HTML section **“Implied vs realized”** (feature importance callout + model
horse-race row(s) with bootstrap gates)
- Methodology + package docs; mark M9 in `plan.md` when exit met
- Unit tests (network-free) for builders, unit conversion, model fit/predict,
registry wiring, and leakage



### Stretch (same milestone if schedule allows; else M10 polish)

- Modest cross-asset expansion via yfinance only, e.g. Treasury yield proxy
(`^TNX` or equivalent) and/or one peer ETF return/RV feature, behind the
existing `_asof_align` pattern
- CLI `--with` tokens for the stretch family (e.g. `rates`)
- Multi-horizon flagship run with IV−RV + `vix_as_forecast` under
`screen-horizons` (Skill by horizon includes the IV model)



### Out of scope

- Options-implied surfaces / single-name IV vendors (Polygon / similar)
- Option Greeks, variance-swap replication, or pricing lab work
- Intraday / high-frequency RV or true tick bipower
- Parametric OU / stochastic-vol filter baselines (planned M10)
- Granger causality / mutual information as primary claims
- Cross-sectional / portfolio-of-names models
- Live scheduling / production monitoring
- Hyperparameter search / Optuna
- Rewriting walk-forward, inference, or multi-horizon orchestration

---



## Acceptance Criteria

1. An IV−RV feature family is registered and builds at least the locked gap
  columns (see Research Contract) when VIX OHLCV is available.
2. Unit alignment between VIX and `rv_cc_*` is **documented and tested**
  (synthetic panels with known scales).
3. Leakage unit tests assert: VIX as-of ≤ *t*; trailing RV windows end at *t*;
  no use of `target_rv_cc_*` inside gap builders.
4. Model registry includes a VIX/IV proxy forecast (name locked below) that
  implements the same fit/predict surface as other horse-race models.
5. Flagship SPY screen (h=5) with VIX + IV−RV includes the new model in the
  horse-race; M7 block bootstrap vs `har_rv_ols` is reported for it.
6. HTML memo gains an **Implied vs realized** section: gap-feature importance
  (when screened) + IV-model ΔQLIKE / bootstrap CI / p with locked wording.
7. CLI supports building/screening with the new extras without breaking
  existing `--with vix` / `--with jump` behavior.
8. `docs/research_methodology.md` documents IV proxy caveats (VIX ≠ single-name
  IV; unit conversion; daily approximation).
9. Full pytest suite green; `plan.md` gains Milestone 9 DONE when criteria met.
10. **(Stretch)** Rates/peer family and/or multi-horizon IV horse-race shipped
  with leakage tests and docs.

---



## Locked Research Defaults


| Setting                        | Value                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------- |
| Flagship symbol                | SPY                                                                               |
| Batch symbols (optional)       | SPY, QQQ                                                                          |
| Primary horizon                | **5** trading days (multi-horizon optional stretch)                               |
| Target                         | `target_rv_cc_5d` (or `target_rv_cc_{h}d` when multi-horizon)                     |
| Primary metric                 | QLIKE (lower better)                                                              |
| Secondary metrics              | MSE, MAE (descriptive)                                                            |
| Walk-forward                   | expanding, `n_splits=5`, `embargo_size = horizon_days`                            |
| Inference baseline             | `har_rv_ols`                                                                      |
| Horse-race models (min)        | `har_rv_ols`, `ridge`, `lasso`, `vix_as_forecast`                                 |
| Primary inference              | Moving block bootstrap of mean OOS ΔQLIKE (M7/M8)                                 |
| Secondary inference            | Optional HLN–DM + NW (`nw_lags = h − 1`)                                          |
| `alpha`                        | **0.05**                                                                          |
| Significance wording           | “Significantly better” **only** if bootstrap rejects at α **and** mean ΔQLIKE < 0 |
| IV proxy                       | VIX (existing ingest / Parquet path; yfinance)                                    |
| Feature family name            | `iv_rv`                                                                           |
| Core gap windows               | **1 / 5 / 21** (align with HAR)                                                   |
| CLI core extras                | `--with vix` still loads VIX; `--with iv_rv` (or `vix,iv_rv`) enables gap family  |
| Artifact root (single-horizon) | existing screen artifact layout under `data/artifacts/`                           |




### Unit conversion (locked)

Platform trailing RV (`rv_cc_*`) is **non-annualized** close-to-close volatility
over the trailing window (see features README). VIX prints are conventionally
**annualized percent** (e.g. `20` ≈ 20%).

Lock one conversion for all gap features and for `vix_as_forecast`:

```text
vix_vol_daily = (vix_level / 100.0) / sqrt(252)
```

Then:

```text
vix_minus_rv_{w}d = vix_vol_daily − rv_cc_{w}d
```

Rationale: put both series on the same daily-vol scale as the target family.
Do **not** invent a second competing conversion in code paths; if research later
wants annualized gaps, add an explicit alternate family — do not silently mix.

Document in methodology: this is a **research proxy**, not a variance-swap or
options-replication identity.

### Why these defaults


| Choice                       | Rationale                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------- |
| VIX as IV proxy              | Already ingested; liquid; matches SPY/index-ETF flagship without new vendors |
| Gap at HAR windows 1/5/21    | Comparable to existing HAR structure; interpretable per horizon memory       |
| `vix_as_forecast` as a model | Separates “IV as covariate” from “IV as competing forecast”                  |
| Keep HAR as baseline         | Design principle: sophisticated / alternative forecasts must beat HAR        |
| h=5 primary                  | Matches flagship demo; M8 already covers horizon dimension                   |
| Stretch rates/peer           | Useful, but secondary to the implied-vs-realized claim                       |


---



## Target Folder Additions

```text
src/vip/
  features/
    cross_asset.py              # EDIT: reuse _asof_align; optional rates stretch
    iv_rv_features.py           # NEW: gap builders + unit helpers
    pipeline.py                 # EDIT: append iv_rv when extras enabled
    registry.py                 # EDIT: register family "iv_rv"
    README.md
  modeling/
    baselines.py                # EDIT or NEW module: VixAsForecastModel
    registry.py                 # EDIT: register vix_as_forecast
    README.md
  application/
    build_feature_matrix.py     # EDIT: FeatureMatrixExtras.include_iv_rv
    screen_factors.py           # EDIT: include vix_as_forecast in HORSE_RACE_MODELS
    screen_multi_horizon.py     # EDIT only if stretch multi-horizon IV path
  reporting/
    experiment_summary.py       # EDIT: Implied vs realized section inputs
    templates/
      factor_screen.html.j2     # EDIT: Implied vs realized section
  cli/
    feature_extras.py           # EDIT: allow token iv_rv (+ rates stretch)
    commands/features.py        # thin; extras via shared parser
    commands/screen.py          # no parallel stack
    README.md

docs/
  research_methodology.md       # IV−RV / VIX-as-forecast section
  milestones/
    milestone 9 walkthrough.md

tests/unit/
  test_iv_rv_features.py        # builders, units, leakage
  test_vix_as_forecast.py       # fit/predict, missing cols, floor
  test_iv_rv_screen_wiring.py   # optional: horse-race includes new model (tmp_path)
```

Reuse — do **not** invent a parallel evaluation stack: extend registries and
call existing `screen_factors` / M7 inference.

---



## Research Contract



### IV−RV feature matrix columns (core)

When `include_iv_rv=True` and VIX OHLCV is present, append:


| Column             | Definition (information ≤ *t*)                    |
| ------------------ | ------------------------------------------------- |
| `vix_vol_daily`    | `(vix_level / 100) / sqrt(252)` after as-of align |
| `vix_minus_rv_1d`  | `vix_vol_daily − rv_cc_1d`                        |
| `vix_minus_rv_5d`  | `vix_vol_daily − rv_cc_5d`                        |
| `vix_minus_rv_21d` | `vix_vol_daily − rv_cc_21d`                       |


Optional (same milestone if cheap; otherwise skip):


| Column            | Definition                                      |
| ----------------- | ----------------------------------------------- |
| `vix_rv_ratio_5d` | `vix_vol_daily / rv_cc_5d` with zero/NaN guards |


Keep existing `vix_level` / `vix_chg_1d` under the VIX path (`include_vix`).
Do **not** drop them when enabling `iv_rv`.

### Builder rules

1. Require HAR columns `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d` already on the
  primary frame (build after HAR family, or accept them as inputs).
2. Align VIX with existing `build_vix_features` / `_asof_align` (backward
  as-of). Prefer composing on `vix_level` rather than re-fetching.
3. Centralize conversion in one helper, e.g. `vix_level_to_daily_vol(series)`.
4. Drop / propagate NaNs consistently with the feature pipeline (early rows
  NaN until windows and first VIX print exist).
5. Public APIs ≤5 parameters (bundle options in a small dataclass if needed).



### Competing forecast: `vix_as_forecast`

**Role.** A simple baseline that maps as-of VIX into the forward-RV units used
by the target — not a full factor model.

**Locked behavior:**

1. Uses column `vix_vol_daily` if present; otherwise derives it from
  `vix_level` via the locked conversion.
2. `fit`: estimate a univariate mapping on the training fold only, e.g.
  OLS of `target` on `vix_vol_daily` + intercept (same spirit as HAR OLS),
   **or** a no-intercept scale if you lock that instead — pick one and test it.
   Preferred default: **intercept OLS on** `vix_vol_daily` (transparent,
   comparable to HAR’s OLS style).
3. `predict`: apply fitted params; floor predictions at the same positive
  floor used by other baselines (`1e-8` unless a shared constant already
   exists).
4. Must raise typed `DataValidationError` on missing columns / empty train /
  unfitted predict.
5. Register as `vix_as_forecast` in `create_default_model_registry()`.
6. Add to `HORSE_RACE_MODELS` (or screen config equivalent) so inference runs
  automatically vs `har_rv_ols`.

**Non-goals for this model:** using the full IV−RV gap vector; tree methods;
hyperparameter search. Gaps belong in the **feature** screen (Ridge/Lasso/
importance), not inside this baseline.

### Screen + inference

For the flagship h=5 run with `--with vix,iv_rv` (exact token spelling locked
in CLI section):

1. Build/load feature matrix including VIX + IV−RV columns.
2. Run horse-race including `vix_as_forecast`.
3. Persist OOS losses; run block bootstrap ΔQLIKE vs `har_rv_ols` for each
  challenger (including the VIX forecast).
4. Factor importance on the screening model (typically Ridge) may rank
  `vix_minus_rv_*` — report top gaps in the new HTML section when present.
5. Wording: never claim “IV beats HAR” from point QLIKE alone.



### CLI extras

Extend `parse_feature_extras`:


| Token   | Effect                                                                                                                                                                           |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vix`   | Existing: ingest/load VIX; append `vix_level`, `vix_chg_1d`                                                                                                                      |
| `jump`  | Existing: jump proportion family                                                                                                                                                 |
| `iv_rv` | **New:** enable IV−RV family; **implies** VIX must be available (auto-set `include_vix=True` or require `vix` alongside — pick one; prefer auto-imply with a clear log/doc note) |
| `rates` | Stretch only                                                                                                                                                                     |


Backward compatible: `--with vix` alone must behave as today (level/chg only,
unless you explicitly document that `vix` also builds gaps — **prefer keeping
gaps behind** `iv_rv` so ablation stays possible).

### Artifacts / report


| Artifact / section           | Content                                                              |
| ---------------------------- | -------------------------------------------------------------------- |
| Feature matrix columns       | Include `vix_vol_daily`, `vix_minus_rv_*` when enabled               |
| `metrics.json` / horse-race  | Row for `vix_as_forecast`                                            |
| `inference.json`             | Bootstrap CI/p for `vix_as_forecast` vs HAR                          |
| HTML **Implied vs realized** | Unit caveat; gap importance (if any); IV-model ΔQLIKE + significance |
| Methodology                  | VIX proxy limits; conversion formula; daily not options-replication  |


---



## Design Rules

1. **Reuse** VIX as-of joins, feature registry, model registry, `screen_factors`,
  and M7 inference — M9 is features + one baseline + reporting.
2. **One conversion helper** for VIX → daily vol; no duplicated magic factors.
3. **Gaps are features; VIX-as-forecast is a model** — keep the research
  questions separable in the report.
4. **Do not export redundant level columns** that duplicate HAR (lesson from
  jump/`bpv_cc_*`): prefer gaps / converted VIX daily vol, not a second RV
   clone.
5. Bootstrap remains **primary**; HLN–DM secondary; wording unchanged from M7.
6. CLI thin; math in `features` / `modeling`; orchestration stays in
  `application`.
7. NumPy docstrings; module/class docs; ≤5 params (bundle options); no broad
  `except`; typed domain errors; no unused typing imports.
8. Avoid magic numbers in array indexing; name window constants
  (`WINDOW_1D`, etc.).
9. Do not mutate caller frames inside helpers.
10. Stretch rates/peer must pass leakage tests; no new market-data vendor.

---



## Step-by-Step Build Plan



### Step 1 — Unit conversion helper + IV−RV builders

Add `iv_rv_features.py` (or extend `cross_asset.py` if that keeps imports
cleaner — prefer a dedicated module for clarity):

- `vix_level_to_daily_vol`
- `build_iv_rv_features(primary_features, vix_ohlcv | vix_level_series)`
- Locked output columns

Unit tests with synthetic VIX and RV series (known arithmetic).  
**Checkpoint:** `test_iv_rv_features.py` green for units + shapes.

### Step 2 — Leakage tests

Assert:

- Permuting future VIX prints does not change features at *t* (as-of
discipline), or equivalent merge_asof contract tests already used for VIX.
- Gap at *t* depends on `rv_cc_`* ending at *t* only.
- Target column never read by the builder.

**Checkpoint:** leakage cases in `test_iv_rv_features.py` green.

### Step 3 — Registry + pipeline + FeatureMatrixExtras

- Register family `iv_rv`.
- Thread `include_iv_rv` through `FeatureMatrixExtras` / `build_feature_matrix`
/ pipeline.
- CLI: `--with iv_rv` (imply VIX).

**Checkpoint:** `vip features --help` shows token; unit test builds matrix with
IV−RV columns on a tiny fixture.

### Step 4 — `vix_as_forecast` model

Implement model + registry entry + unit tests (fit/predict/floor/errors).  
**Checkpoint:** `test_vix_as_forecast.py` green.

### Step 5 — Horse-race wiring

Add `vix_as_forecast` to screen horse-race models; confirm inference artifacts
include it. Keep param counts clean via existing config objects.  
**Checkpoint:** wiring test or small screen on synthetic panel includes the
model name in summary/inference.

### Step 6 — HTML + wording

Add **Implied vs realized** section:

- Short caveat (VIX proxy; unit conversion).
- Table/rows for IV model vs HAR (ΔQLIKE, CI, p, significant flag).
- Optional: top IV−RV features by permutation importance.

**Checkpoint:** template/report test asserts section presence.

### Step 7 — Methodology + package docs + [plan.md](http://plan.md)

Update `docs/research_methodology.md`, features/modeling/cli READMEs, and add
Milestone 9 section to `plan.md` (status → DONE only at exit).  
**Checkpoint:** docs match locked formula and model name.

### Step 8 — Flagship run + exit

```powershell
$env:PYTHONPATH = "src"
py -m pip install -e ".[dev]"
vip ingest --symbol SPY
vip ingest --symbol VIX
vip features --symbol SPY --with vix,iv_rv
vip screen --symbol SPY
# inspect report.html → Implied vs realized
py -m pytest -q
```

Optional stretch:

```powershell
vip screen-horizons --symbol SPY --with vix,iv_rv
# and/or --with vix,iv_rv,rates once stretch exists
```

**Checkpoint:** acceptance criteria met; `plan.md` M9 DONE.

### Step 9 — Stretch (optional)

Rates/peer family + leakage tests + CLI token; and/or multi-horizon IV
horse-race narrative in the study HTML.

---



## Suggested Command Sequence

```powershell
$env:PYTHONPATH = "src"
py -m pip install -e ".[dev]"

py -m pytest tests/unit/test_iv_rv_features.py -q
py -m pytest tests/unit/test_vix_as_forecast.py -q

vip ingest --symbol SPY
vip ingest --symbol VIX
vip features --symbol SPY --horizon 5 --with vix,iv_rv
vip screen --symbol SPY
vip run --symbol SPY --with vix,iv_rv

# inspect data/artifacts/**/report.html (Implied vs realized)
py -m pytest -q
```

Optional multi-horizon stretch:

```powershell
vip screen-horizons --symbol SPY --with vix,iv_rv
```

---



## Common Pitfalls

- Subtracting raw VIX (percent annualized) from non-annualized `rv_cc_*`
without conversion — gaps become meaningless and models look “significant”
for the wrong reason.
- Building gaps from **forward** RV / target columns (leakage).
- Registering `vix_as_forecast` but leaving it out of `HORSE_RACE_MODELS`, so
the report never shows inference.
- Folding the entire gap vector into the “IV forecast” model — blurs feature
vs model questions.
- Re-introducing duplicate vol **level** columns that destabilize permutation
importance (same failure mode as exporting `bpv_cc_*`).
- Claiming “implied beats realized” from a QLIKE ranking without bootstrap
gates.
- Breaking `--with vix` / `--with jump` while adding `iv_rv`.
- Pulling a new data vendor for single-name IV “just for M9.”
- Rewriting multi-horizon orchestration instead of reusing `screen_factors`.

---



## Decisions Locked for This Walkthrough

1. **M9 theme = IV−RV gap & implied-as-forecast** on the existing VIX path.
2. Unit conversion: `vix_vol_daily = (vix_level / 100) / sqrt(252)`.
3. Core gap columns at windows **1 / 5 / 21**; family name `iv_rv`.
4. Competing model name: `vix_as_forecast` (univariate OLS on daily VIX vol
  - intercept, positive prediction floor).
5. Inference baseline remains `har_rv_ols`; M7 bootstrap primary.
6. Primary flagship horizon remains **5**; multi-horizon IV study is stretch.
7. CLI: gaps behind `iv_rv` (implies VIX); do not silently change bare
  `--with vix` column set.
8. Stretch: modest rates/peer cross-asset only; OU/SV baselines wait for M10.
9. Out of scope: options surfaces, HF RV, Granger/MI as claims, cross-section,
  scheduling.

---



## Milestone 9 Exit Checklist

- [x] `vix_level_to_daily_vol` + IV−RV builders + unit tests
- [x] Leakage tests for IV−RV / as-of discipline
- [x] Registry + pipeline + `FeatureMatrixExtras.include_iv_rv`
- [x] CLI `--with iv_rv` (VIX implied) without regressing `vix` / `jump`
- [x] `vix_as_forecast` model + registry + unit tests
- [x] Horse-race + M7 inference include `vix_as_forecast`
- [x] HTML “Implied vs realized” section + locked wording
- [x] Methodology + READMEs + `plan.md` Milestone 9 section
- [x] Flagship SPY run inspected; full `pytest -q` green; plan status DONE
- [x] (Stretch) rates family (`--with rates`, TNX asof join, leakage tests)
- [ ] (Stretch) peer ETF family (deferred)
- [ ] (Stretch) multi-horizon IV HTML narrative (deferred)


**Acceptance map:** checklist rows 1–7 ↔ criteria 1–7 (code). Row 8 ↔
criteria 8–9 docs half. Row 9 ↔ criteria 9 exit gate. Row 10 ↔ criterion 10
(stretch; not blocking core DONE). Note: `iv_rv` is pipeline-composed (not an
OHLCV `FeatureSpec` in `create_default_registry`); enable via extras/CLI.

---



## What Comes Next (post-M9)

Ordered backlog after implied-vs-realized:

1. **Milestone 10 — Parametric / filter baselines** (discrete OU-style / simple
  SV-inspired filters); must beat HAR on OOS QLIKE to matter
2. Optional diagnostics (Granger, MI, rough-vol memory) as appendix tooling
3. Broader cross-asset covariates once the as-of pattern is proven again on
  rates/peers

Still deferred: intraday HF RV, options surfaces/Greeks, sentiment,
cross-sectional models, live scheduling.