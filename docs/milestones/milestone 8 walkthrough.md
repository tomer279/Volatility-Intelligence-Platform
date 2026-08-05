# Milestone 8 Walkthrough — Multi-Horizon Factor Intelligence

## Objective

Promote horizon from a single config knob to a first-class research dimension:
run the existing factor screen + M7 inference stack across **1d / 5d / 21d**
forward RV targets and report how model and factor skill change with horizon.

This milestone should prove:

- The platform builds and screens **multiple horizons in one study**, not only by
manually flipping `target.horizon_days` once.
- Per horizon, walk-forward uses **horizon-aware embargo**, **NW lags =
horizon − 1**, and **block-bootstrap inference** vs `har_rv_ols` (M7 contract
generalized, not rewritten).
- Artifacts and the HTML memo answer: **what predicts next-day vs next-week vs
next-month RV**, with locked “significantly better” wording still gated on the
primary bootstrap.
- Optional stretch: **jump-robust realized features** (bipower / jump proportion)
plug into the registry and appear in the multi-horizon horse-race.

---

## Scope

### In scope

- Multi-horizon study orchestration (application use-case + CLI) over locked
horizons `{1, 5, 21}`
- Per-horizon feature matrices with `target_rv_cc_{h}d` (reuse
`build_target_rv_cc` / `build_feature_matrix`)
- Per-horizon walk-forward screen: same horse-race models, factor importance,
M7 OOS losses + block bootstrap (+ optional HLN–DM)
- Horizon-scaled validation defaults: `embargo_size`, `nw_lags`,
`bootstrap_block_length` (extend M7’s hard `[10, 20]` range where h=21 needs it)
- Cross-horizon summary artifact + HTML section (“Skill by horizon”)
- Config / CLI: `--horizons 1,5,21` (or YAML list); keep single-horizon path
backward compatible
- Unit tests (network-free): horizon naming, embargo/NW helpers, multi-horizon
artifact shape, no leakage across horizons
- Update `docs/research_methodology.md` and mark M8 in `plan.md` when exit met

### Stretch (same milestone if schedule allows; else M9)

- Jump-robust features in `vip.features.realized` (+ registry family):
bipower variation, jump proportion (daily close-to-close approximation as
documented — not true high-frequency bipower)
- Leakage tests for jump features; optional importance callout in the report

### Out of scope

- Options-implied surfaces / single-name IV vendors
- Intraday / high-frequency RV (true tick bipower)
- Parametric OU / stochastic-vol filter baselines (later backlog)
- Additional cross-asset beyond existing VIX path
- Granger / mutual-information diagnostics
- Cross-sectional / portfolio-of-names models
- Live scheduling / production monitoring
- Hyperparameter search / Optuna
- Rewriting the walk-forward or inference engines

---

## Acceptance Criteria

1. A multi-horizon study runs for SPY over horizons **1, 5, and 21** trading days
  from one command (or one config), producing per-horizon screens.
2. Each horizon uses target column `target_rv_cc_{h}d` and **embargo ≥ h**
  (default `embargo_size = h`).
3. Per horizon, M7 inference runs vs `har_rv_ols` with
  `nw_lags = horizon_days - 1` and block bootstrap; significance wording still
   gated on the primary bootstrap.
4. Bootstrap block length is **horizon-aware** (see locked defaults); config
  validation allows the h=21 default (must not hard-fail on the old `[10, 20]`
   cap alone).
5. Artifacts include a **cross-horizon summary** (JSON + HTML section) comparing
  mean OOS QLIKE and mean ΔQLIKE / bootstrap p vs HAR by model × horizon.
6. Single-horizon `vip screen` / `vip run` remain backward compatible (default
  horizon still **5**).
7. Unit tests cover: target naming for h∈{1,5,21}; `nw_lags_for_horizon`;
  embargo helper; multi-horizon wiring on synthetic panels; no network.
8. `docs/research_methodology.md` documents multi-horizon evaluation and
  horizon-scaled inference defaults.
9. Full pytest suite green; `plan.md` gains Milestone 8 DONE when criteria met.
10. **(Stretch)** Jump-robust feature family registered, leakage-tested, and
  optionally included in the flagship multi-horizon run.

---

## Locked Research Defaults


| Setting                                     | Value                                                                             |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| Flagship symbol                             | SPY                                                                               |
| Batch symbols (optional)                    | SPY, QQQ (reuse `screen-batch` pattern if cheap)                                  |
| Horizons                                    | **1, 5, 21** trading days (locked study set)                                      |
| Primary horizon (legacy single-run default) | **5** (unchanged)                                                                 |
| Target family                               | close-to-close forward RV → `target_rv_cc_{h}d`                                   |
| Primary metric                              | QLIKE (lower better)                                                              |
| Secondary metrics                           | MSE, MAE (descriptive)                                                            |
| Walk-forward                                | expanding, `n_splits=5`                                                           |
| Embargo                                     | `embargo_size = horizon_days` per horizon                                         |
| Inference baseline                          | `har_rv_ols`                                                                      |
| Horse-race models (min)                     | `har_rv_ols`, `ridge`, `lasso` (+ `random_forest` when already in screen)         |
| Primary inference                           | Moving block bootstrap of mean OOS ΔQLIKE (M7)                                    |
| Secondary inference                         | Optional HLN–DM + NW (`nw_lags = h − 1`)                                          |
| `alpha`                                     | **0.05**                                                                          |
| `bootstrap_n_resamples`                     | ≥ 999 (1999 for reports; 999 OK in tests)                                         |
| `bootstrap_random_seed`                     | `0`                                                                               |
| Significance wording                        | “Significantly better” **only** if bootstrap rejects at α **and** mean ΔQLIKE < 0 |
| Feature families (core)                     | returns, har, range, volume (+ VIX/jump when `--with vix` / `jump`)               |
| Artifact root                               | `data/artifacts/multi-horizon-screen-{symbol}-{date}/`                            |


### Per-horizon inference defaults


| Horizon `h` | Target             | `embargo` | `nw_lags` | Default `bootstrap_block_length` | Allowed block range  |
| ----------- | ------------------ | --------- | --------- | -------------------------------- | -------------------- |
| 1           | `target_rv_cc_1d`  | 1         | 0         | **10**                           | 5–15                 |
| 5           | `target_rv_cc_5d`  | 5         | 4         | **15**                           | 10–20 (M7 unchanged) |
| 21          | `target_rv_cc_21d` | 21        | 20        | **21**                           | 15–42                |


Rationale: overlapping *h*-step labels induce dependence of order ~*h*; NW lag
stays `h − 1`; block length tracks horizon so h=21 is not forced into an
under-blocked ℓ=15. Keep helpers centralized (do not scatter magic numbers).

### Why these defaults


| Choice                   | Rationale                                                                        |
| ------------------------ | -------------------------------------------------------------------------------- |
| Horizons 1 / 5 / 21      | Classic short / week / month trading-day set; matches existing HAR lag structure |
| Embargo = h              | Label overlap + feature lookback safety; matches M3–M7 spirit for h=5            |
| Reuse M7 inference       | Multi-horizon is an evaluation *dimension*, not a new significance theory        |
| Cross-horizon summary    | PM-facing answer without opening three unrelated HTML memos as the only view     |
| Jump features as stretch | Enrich predictors without new vendors; true HF bipower stays deferred            |


---

## Target Folder Additions

```text
src/vip/
  application/
    screen_multi_horizon.py   # NEW: orchestrate per-horizon screens + summary
    screen_factors.py         # ensure horizon/embargo/inference params stay injectable
    build_feature_matrix.py   # already horizon-aware; reuse
    run_study.py              # optional: --horizons path or delegate
  evaluation/
    inference.py              # horizon-aware block-length defaults / validation
    comparison.py             # optional: stack or tag summaries with horizon
    __init__.py
    README.md
  features/
    realized.py               # STRETCH: bipower / jump proportion helpers
    jump_features.py          # STRETCH: registry builder family
    registry.py               # STRETCH: register "jump" family
  reporting/
    experiment_summary.py     # cross-horizon table + wording
    templates/
      factor_screen.html.j2   # or multi_horizon_screen.html.j2
  cli/commands/
    screen_multi_horizon.py   # NEW: vip screen-horizons (name TBD)
    screen.py                 # keep single-horizon; document relationship
  config/
    schema.py                 # optional: horizons: list[int]

docs/
  research_methodology.md     # multi-horizon section
  milestones/
    milestone 8 walkthrough.md

tests/unit/
  test_multi_horizon_defaults.py   # embargo, nw_lags, block length per h
  test_screen_multi_horizon.py     # wiring / artifact shape (tmp_path)
  test_jump_features.py            # STRETCH: builders + leakage
```

Reuse — do **not** invent a parallel CV/inference stack: call existing
`screen_factors` (or its internals) once per horizon with injected
`horizon_days`, `embargo_size`, and inference options.

---

## Research Contract

### Per-horizon matrix

For each `h` in `{1, 5, 21}`:

1. Build (or load) a feature matrix whose target column is `target_rv_cc_{h}d`.
2. Predictors remain information set ≤ *t* (existing families; optional VIX).
3. Drop rows with NaN target/features as today.
4. Persist under the study dir, e.g. `h{h}d/feature` lineage via existing stores
  or per-horizon processed keys — prefer reusing `build_feature_matrix` with
   `horizon_days=h`.

### Per-horizon screen + inference

For each horizon matrix:

1. Run the same horse-race models through expanding walk-forward with
  `n_splits=5`, `embargo_size=h`.
2. Collect per-row OOS QLIKE losses; persist `h{h}d/oos_losses.json` (or
  equivalent nested layout).
3. For each challenger vs `har_rv_ols`, compute mean ΔQLIKE, block bootstrap CI /
  p-value with horizon-specific block length; optional HLN–DM with
   `nw_lags = h - 1`.
4. Write `h{h}d/metrics.json`, `h{h}d/inference.json`, importance artifacts,
  and optionally a per-horizon `report.html`.

### Cross-horizon summary

Build one table keyed by `(horizon_days, model)`:


| Column                                   | Role                    |
| ---------------------------------------- | ----------------------- |
| `horizon_days`                           | 1, 5, or 21             |
| `model`                                  | horse-race name         |
| `qlike` / `mse` / `mae`                  | mean OOS metrics        |
| `mean_delta_qlike`                       | vs HAR at that horizon  |
| `bootstrap_ci_low` / `bootstrap_ci_high` | primary CI              |
| `bootstrap_pvalue`                       | primary p               |
| `significant_vs_baseline`                | bootstrap @ α and Δ < 0 |


Persist as `horizon_summary.json` (and optional CSV). HTML gains a **Skill by
horizon** section: compact table + short narrative using locked wording
(never claim significance from a QLIKE ranking alone).

### Embargo / block-length helpers

Centralize defaults (keep public APIs ≤5 params via a frozen options object), e.g.:

```python
def default_embargo_for_horizon(horizon_days: int) -> int:
    """Return embargo_size = horizon_days."""

def default_bootstrap_block_length(horizon_days: int) -> int:
    """Return locked default block length for horizon_days."""

def validate_bootstrap_block_length(horizon_days: int, block_length: int) -> None:
    """Validate block_length against the horizon-specific allowed range."""
```

`nw_lags_for_horizon` already returns `horizon_days - 1` — reuse it; do not
special-case h=5 only.

### Jump-robust features (stretch)

Daily approximation (document limitations vs true HF bipower):

- Trailing bipower variation / bipower vol over locked windows (align with HAR
windows 1 / 5 / 21 where practical).
- Jump proportion proxy: max(0, RV − bipower) / RV (guard zeros).

Registry family name e.g. `jump`. Leakage: trailing only, no forward windows.
Methodology must state these are **daily** proxies, not tick-based estimators.

### Artifacts written


| Artifact                       | Content                                                  |
| ------------------------------ | -------------------------------------------------------- |
| `screen_meta.json`             | horizons, per-h embargo / NW / block length, models, α   |
| `h{h}d/metrics.json`           | per-horizon horse-race + inference                       |
| `h{h}d/oos_losses.json`        | per-row OOS losses                                       |
| `h{h}d/inference.json`         | challenger inference rows                                |
| `h{h}d/importance.json` (etc.) | existing screen artifacts                                |
| `horizon_summary.json`         | cross-horizon comparison table                           |
| `report.html`                  | study memo with Skill-by-horizon + optional per-h detail |


---

## Design Rules

1. **Reuse** `build_feature_matrix`, `screen_factors` / walk-forward collectors,
  and M7 `inference` — multi-horizon is orchestration + reporting.
2. **One horizon per walk-forward run** — do not mix `target_rv_cc_1d` and
  `target_rv_cc_21d` labels in the same fold metrics without tagging.
3. **Embargo ≥ horizon** for each run; default equality `embargo = h`.
4. **NW lag = h − 1** on every horizon (including 0 when h=1).
5. **Bootstrap remains primary**; HLN–DM secondary; wording unchanged from M7.
6. **Horizon-aware block length** — lift or parameterize the old global
  `[10, 20]` validator so h=21 defaults are legal.
7. CLI thin; math stays in `evaluation`; orchestration in `application`.
8. NumPy docstrings; module/class docs; ≤5 params (bundle options); no broad
  `except`; typed domain errors.
9. Do not mutate caller frames inside helpers.
10. Stretch jump features must pass leakage tests; no HF data dependency.

---

## Step-by-Step Build Plan

### Step 1 — Horizon default helpers

Add `default_embargo_for_horizon`, `default_bootstrap_block_length`, and
horizon-specific block-length validation (extend `inference.py` or a tiny
`evaluation/horizon_defaults.py` if that keeps param counts clean).

Unit test the locked table for h∈{1,5,21}.  
**Checkpoint:** `test_multi_horizon_defaults.py` green.

### Step 2 — Confirm single-horizon path is injectable

Audit `screen_factors` / `FactorScreenConfig` (or equivalent): ensure
`horizon_days`, `embargo_size`, target column, and inference options can be set
per call without relying on process-global config only.

Fix any hard-coded `target_rv_cc_5d` / `embargo=5` that would break h≠5.  
**Checkpoint:** unit or smoke: screen with `horizon_days=1` writes
`target_rv_cc_1d` metrics.

### Step 3 — Multi-horizon orchestrator

Implement `screen_multi_horizon`:

1. Validate horizons list (default `[1, 5, 21]`).
2. For each h: build/load matrix → run screen with horizon defaults → write
  under `h{h}d/`.
3. Stack metrics into `horizon_summary.json`.
4. Return a structured result object for CLI/reporting.

**Checkpoint:** `test_screen_multi_horizon.py` on tiny synthetic frames (tmp_path).

### Step 4 — CLI

Add `vip screen-horizons` (or `vip screen --horizons 1,5,21`):

- Flags: `--symbol`, `--horizons`, `--with` (`vix`,`jump`), `--skip-features`.
- Print a compact cross-horizon QLIKE / ΔQLIKE / p table.
- Echo artifact root.

Keep `vip screen` as the single-horizon entrypoint (default h=5).  
**Checkpoint:** `--help` + dry wiring test.

### Step 5 — HTML + wording

Extend reporting:

- Study-level memo with **Skill by horizon** table.
- Per-horizon caveats: embargo, NW lags, block length, α, baseline.
- Reuse M7 significance language per cell / row.

**Checkpoint:** HTML test asserts new section / columns exist.

### Step 6 — Methodology + package docs

Update `docs/research_methodology.md` (multi-horizon evaluation; per-h inference
defaults; relationship to M7). Update `evaluation` / `application` READMEs.  
**Checkpoint:** docs match code defaults.

### Step 7 — Stretch: jump-robust features

Implement daily bipower / jump proportion helpers + registry family + leakage
tests; CLI opt-in via `--with jump` on features / run / screen-horizons.
**Checkpoint:** tests green; methodology notes daily-proxy limitation.

### Step 8 — Flagship run + plan status

Run SPY multi-horizon study (with VIX if desired); spot-check
`horizon_summary.json` and `report.html`. Mark M8 DONE in `plan.md` when
acceptance criteria met.  
**Checkpoint:** full `pytest -q` green.

---

## Suggested Command Sequence

```powershell
$env:PYTHONPATH = "src"
py -m pip install -e ".[dev]"
py -m pytest tests/unit/test_multi_horizon_defaults.py -q
py -m pytest tests/unit/test_screen_multi_horizon.py -q
vip features --symbol SPY --horizon 1 --with vix
vip features --symbol SPY --horizon 5 --with vix
vip features --symbol SPY --horizon 21 --with vix
vip screen-horizons --symbol SPY --with vix
# inspect data/artifacts/multi-horizon-screen-spy-*/horizon_summary.json and report.html
py -m pytest -q
```

Optional stretch:

```powershell
py -m pytest tests/unit/test_jump_features.py -q
vip screen-horizons --symbol SPY --with vix,jump
```

---

## Common Pitfalls

- Running three horizons but leaving **embargo=5** for h=21 (label leakage risk /
under-embargo).
- Leaving **NW lags = 4** for every horizon instead of `h − 1`.
- Keeping a global bootstrap **block_length ∈ [10, 20]** validator that rejects
the h=21 default.
- Mixing horizons in one metrics table **without** a `horizon_days` key.
- Claiming “HAR dominates short horizons” from **point QLIKE** without bootstrap
gates (M7 wording still applies per horizon).
- Rebuilding a **second** walk-forward/inference stack instead of looping
`screen_factors`.
- Treating stretch **daily bipower** as true high-frequency jump measurement in
the report narrative.
- Breaking single-horizon `vip screen` / default h=5 demos while adding
multi-horizon.

---

## Decisions Locked for This Walkthrough

1. **M8 theme = multi-horizon factor intelligence** over locked horizons
  **1 / 5 / 21**.
2. Per horizon: **embargo = h**, **nw_lags = h − 1**, block bootstrap primary vs
  `har_rv_ols`, HLN–DM secondary.
3. Default block lengths: **10 / 15 / 21** for h = 1 / 5 / 21; validation ranges
  as in the per-horizon table.
4. Cross-horizon `horizon_summary.json` **+ HTML section** required for exit.
5. Single-horizon default remains **5-day**; existing CLI paths stay valid.
6. Jump-robust daily features are **stretch**, not required for M8 exit.
7. Out of scope: HF RV, options surfaces, parametric vol baselines, Granger/MI,
  cross-section, scheduling.

---

## Milestone 8 Exit Checklist

- [x] Horizon default helpers (embargo, block length, validation) + tests
- [x] Single-horizon screen works for h∈{1,5,21} with correct target column
- [x] `screen_multi_horizon` writes per-horizon artifacts under one study root
- [x] M7 inference wired per horizon (ΔQLIKE, bootstrap CI/p, optional HLN–DM)
- [x] `horizon_summary.json` + HTML “Skill by horizon” section
- [x] CLI entrypoint for multi-horizon study
- [x] Methodology + README updates
- [x] Full pytest green; `plan.md` M8 DONE
- [ ] (Stretch) jump-robust feature family + leakage tests + optional CLI flag

---

## What Comes Next (post-M8)

Ordered backlog after multi-horizon:

1. Jump-robust / richer realized estimators (if not finished as M8 stretch)
2. Stronger IV−RV / additional cross-asset covariates
3. Parametric / filter baselines in the same horse-race
4. Optional diagnostics (Granger, MI, rough-vol memory)

Still deferred: intraday HF RV, options surfaces/Greeks, sentiment,
cross-sectional models, live scheduling.