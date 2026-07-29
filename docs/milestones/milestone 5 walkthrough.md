# Milestone 5 Walkthrough — Nonlinear & Robustness

## Objective

Extend the M4 factor-screening platform from linear models on own-symbol features to a robustness story: nonlinear models, SHAP-style attribution, VIX/cross-asset covariates, and regime-sliced evaluation (“what works when”).

This milestone should prove:

- Tree models (RF + optional LightGBM) implement the same `fit` / `predict` contract and plug into `run_walk_forward`.
- SHAP (or TreeSHAP) attribution is computed **after** train-only fits, on held-out rows, without leaking test labels into fitting.
- VIX (and optional peer ETF) features join on the primary calendar with strict `timestamp ≤ t` rules and leakage tests.
- Walk-forward metrics (and optionally importance) can be sliced by locked regimes (COVID stress, 2022 bear).
- Optional multi-symbol batch (SPY / QQQ / IWM) produces comparable horse-race + ranking tables.
- The HTML report gains a **“What works when”** section.

---



## Scope



### In scope

- `vip.modeling.tree_models` — `RandomForestVolModel`; optional `LightGBMVolModel` behind an extra
- Wire trees into `create_default_model_registry` (RF always; LightGBM if import succeeds or via optional dep)
- `vip.evaluation.shap_importance` — fold-wise mean |SHAP| (TreeExplainer); optional path if `shap` installed
- Harden M4 permutation importance aggregation (median / cap) for QLIKE spike robustness
- `vip.features.cross_asset` — VIX level / ΔVIX (and optional peer RV) joined to primary index
- Extend feature pipeline / application build to load auxiliary OHLCV from the Parquet store
- `vip.evaluation.regimes` — named date windows + slice metrics (and optional importance) by regime
- Application: extend `screen_factors` (or add `screen_robustness.py`) for nonlinear + regimes + SHAP artifacts
- Reporting: template section “What works when” + optional SHAP bar plot
- CLI: `vip screen` flags for `--model`, `--with-vix`, `--regimes`; optional `vip screen-batch`
- Unit tests (network-free); leakage tests for cross-asset alignment and SHAP fit isolation
- Deps: `scikit-learn` (already); optional extras `[nonlinear]` → `lightgbm`, `shap`



### Out of scope

- Hyperparameter grids / Optuna
- Diebold–Mariano / formal inference
- Intraday RV, options surfaces
- FastAPI / scheduling (M6+)
- Replacing Ridge as the default *linear* screening model (keep Ridge; trees are a second path)

---



## Acceptance Criteria

1. `RandomForestVolModel` (+ optional LightGBM) unit-tested; prediction floor `1e-8`; train-only fit.
2. Walk-forward horse-race includes at least: `har_rv_ols`, `ridge`, `random_forest` on SPY.
3. SHAP summary path works when `shap` is installed; tests skip cleanly when missing (`pytest.importorskip`).
4. VIX features appear in the feature matrix when aux data is present; leakage test fails if future VIX is used.
5. Regime slices (at least COVID + 2022) produce per-regime QLIKE tables in artifacts + HTML.
6. Permutation importance aggregation supports **median** (and optional per-fold cap) so a single QLIKE spike (e.g. `rv_cc_1d`) cannot dominate ranks.
7. `vip screen --symbol SPY --with-vix` (or equivalent) writes report with “What works when”.
8. Full pytest green; no network in unit tests.

---



## Locked Research Defaults


| Setting                  | Value                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Flagship symbol          | SPY                                                                                                                |
| Batch symbols (optional) | SPY, QQQ, IWM                                                                                                      |
| Target                   | `target_rv_cc_5d`                                                                                                  |
| Primary metric           | QLIKE (lower better)                                                                                               |
| Secondary                | MSE, MAE                                                                                                           |
| Walk-forward             | expanding, `n_splits=5`, `embargo=5`                                                                               |
| Linear screening model   | Ridge (unchanged from M4)                                                                                          |
| Nonlinear default        | RandomForest (`n_estimators=200`, `max_depth=4`, `random_state=0`, `min_samples_leaf=5`)                           |
| LightGBM                 | optional; conservative defaults (`num_leaves=15`, `min_data_in_leaf=20`, `learning_rate=0.05`, `n_estimators=200`) |
| Cross-asset              | `^VIX` via yfinance → store as symbol `VIX` (normalize ticker in config)                                           |
| VIX features             | `vix_level`, `vix_chg_1d` (close-to-close change); no forward fills from the future                                |
| Regimes                  | see table below                                                                                                    |
| Importance (permutation) | primary aggregate = **median** ΔQLIKE across repeats×folds; report mean as secondary; optional `importance_cap`    |
| SHAP                     | mean |SHAP| on **test** rows per fold; aggregate median across folds                                               |
| Artifact root            | `data/artifacts/robust-screen-{symbol}-{date}/` (or reuse factor-screen with suffix)                               |




### Locked regimes


| Name          | Start      | End        | Intent              |
| ------------- | ---------- | ---------- | ------------------- |
| `covid_crash` | 2020-02-20 | 2020-04-30 | Spike / crash vol   |
| `bear_2022`   | 2022-01-03 | 2022-10-14 | Persistent risk-off |
| `full_sample` | matrix min | matrix max | Reference (always)  |


Dates are session dates on the feature index; empty slices → skip with a warning in the report, not a crash.

---



## Target Folder Additions

```text
src/vip/
  modeling/
    tree_models.py           # RF (+ optional LightGBM) adapters
    registry.py              # register tree models
  features/
    cross_asset.py           # VIX / peer joins
    pipeline.py              # accept optional AuxiliaryFrames
  evaluation/
    importance.py            # median / cap aggregation options
    shap_importance.py       # TreeSHAP fold helper
    regimes.py               # RegimeWindow + slice_by_regime
  application/
    screen_factors.py        # extend OR screen_robustness.py
    build_feature_matrix.py  # load VIX aux when requested
  visualization/
    importance_plots.py      # reuse for SHAP bars (or shap_plots.py)
  reporting/
    templates/
      factor_screen.html.j2  # add “What works when” (+ SHAP block)
    experiment_summary.py
  cli/commands/
    screen.py                # new flags
    # optional: screen_batch.py

configs/
  default.yaml               # optional: cross_asset.vix_symbol: VIX
  experiments/
    robust_screen_spy.yaml   # optional

tests/unit/
  test_tree_models.py
  test_cross_asset.py
  test_cross_asset_leakage.py
  test_shap_importance.py      # importorskip("shap")
  test_regimes.py
  test_importance_aggregation.py  # median vs mean spike
  test_screen_robustness.py      # or extend test_screen_factors.py
```

---



## Research Contract



### Tree models

1. `fit(X_train, y_train)`: drop NaN rows jointly; fit sklearn/LightGBM on raw (or optionally scaled — **prefer unscaled trees**; document choice).
2. `predict(X)`: predict; clip `yhat = max(yhat, 1e-8)`.
3. Expose `feature_names_` after fit.
4. Same public surface as Ridge/HAR so `run_walk_forward` needs no fork.

Prefer one internal helper (e.g. `TreeVolModel`) parameterized by estimator factory + thin wrappers — keeps ≤5 public params.

### SHAP

For each fold:

1. Fit model on train only.
2. Build `TreeExplainer` on the fitted model (no test labels).
3. Compute SHAP values on **test** `X` only.
4. Per-feature importance_fold = mean(|SHAP|, axis=0).
5. Aggregate across folds with **median** (aligned with hardened permutation policy).

Do **not** claim SHAP is causal. Report as complementary to permutation ΔQLIKE.

### Cross-asset / VIX

1. Ingest `VIX` (yfinance `^VIX`) into `data/raw/VIX/` via existing `vip ingest --symbol VIX` (adapter maps display symbol ↔ Yahoo ticker if needed).
2. `build_cross_asset_features(primary_index, vix_ohlcv) -> DataFrame` indexed like primary:
  - `vix_level` = VIX close as-of session `t` (reindex with `ffill` only **backward in time**, never peek ahead; prefer `reindex(..., method=None)` then left-join and dropna, or `merge_asof` with `direction="backward"`).
  - `vix_chg_1d` = pct or log change of VIX close using only ≤ t.
3. Concatenate onto own-symbol features **before** final `dropna`.
4. Leakage test: shift VIX forward by 1 day → importance/correlation with target must not improve vs correct alignment in a controlled synthetic check; or assert feature at `t` equals VIX close at `t`, not `t+1`.



### Regime slices

Given a long prediction frame or fold-metric table with a DatetimeIndex:

1. Assign each test row (or fold test span) to zero or more regimes by date overlap.
2. Recompute QLIKE/MSE/MAE **within** each regime’s test rows (pool predictions across folds that land in the window).
3. Persist `metrics_by_regime.json` and render HTML table.

Horse-race models for regime table: `har_rv_ols`, `ridge`, `random_forest` (minimum).

### Importance spike policy (M4 follow-up)

QLIKE permutation Δ can spike for highly collinear HAR lags (e.g. `rv_cc_1d`). For M5:

1. Extend `ImportanceOptions` (or `StabilityOptions`) with:
  - `aggregate: Literal["median", "mean"] = "median"`
  - `delta_cap: float | None = None` — clip each repeat’s ΔQLIKE to `[-cap, +cap]` before aggregate (optional; recommend a high cap or `None` initially, enable if needed).
2. `summarize_importance` ranks by `median_importance` when aggregate is median; still report `mean_importance` as a column.
3. Optional secondary: permutation importance under **MSE** for a sanity rank correlation footnote (not required for exit if time-boxed).

Keep Ridge as the default permutation screening model; trees use SHAP as primary attribution.

---



## Design Rules

1. Reuse `run_walk_forward` / `generate_expanding_folds` — no second CV engine.
2. Optional deps (`lightgbm`, `shap`) must not break core install; guard imports; registry omits LightGBM if missing.
3. CLI thin; orchestration in application; math in evaluation/modeling/features.
4. NumPy docstrings; module/class docs; ≤5 params (bundle in frozen dataclasses).
5. No broad `except`; typed domain errors; for optional imports use `except ImportError` only.
6. Leakage tests are mandatory for cross-asset and for “scaler/model/explainer never sees test y”.
7. Do not mutate caller DataFrames in permutation/SHAP helpers.

---



## Step-by-Step Build Plan



### Step 1 — Optional deps + tree model skeleton

Add optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
nonlinear = [
    "lightgbm>=4.0",
    "shap>=0.44",
]
```

Implement `modeling/tree_models.py` with `RandomForestVolModel` first.  
Tests: shapes; signal recovery vs noise; prediction floor; unfitted predict raises.  
**Checkpoint:** `test_tree_models.py` green without lightgbm/shap.

### Step 2 — Registry + walk-forward smoke

Register `random_forest` in `create_default_model_registry`.  
Synthetic walk-forward: RF should beat historical mean on QLIKE when `y` depends on nonlinear interaction.  
**Checkpoint:** plugs into existing harness.

### Step 3 — Harden permutation aggregation

Add median (+ optional cap) to importance/stability path; unit test where one fold has a huge spike on a noise column — **median rank** keeps signal first, mean rank may not.  
Default `vip screen` ranking switches to median (document in methodology).  
**Checkpoint:** `test_importance_aggregation.py` green.

### Step 4 — Cross-asset features

Implement `cross_asset.py` + pipeline/application wiring.  
Yahoo quirk: ingest symbol `VIX` fetching `^VIX` (small map in yfinance adapter or CLI docs).  
Leakage unit tests. Rebuild SPY features with `--with-vix`.  
**Checkpoint:** matrix contains `vix_level`, `vix_chg_1d`.

### Step 5 — Regime evaluation

`regimes.py` with frozen windows; `metrics_by_regime(y_true, y_pred, index)`; wire into screen use-case.  
**Checkpoint:** JSON + unit tests with synthetic dates.

### Step 6 — SHAP path

`shap_importance.py` with `pytest.importorskip`; RF only for MVP.  
Persist `shap_importance.json` + bar plot.  
**Checkpoint:** tests skip without shap; pass with `[nonlinear]` installed.

### Step 7 — Screen use-case + HTML “What works when”

Extend screen (or new use-case) to:

1. Horse-race: HAR + Ridge + RF (+ LGBM if present).
2. Ridge permutation ranks (median).
3. RF SHAP ranks (if available).
4. Regime metric tables.
5. HTML section comparing full-sample vs COVID vs 2022.

**Checkpoint:** `vip screen --symbol SPY --with-vix` writes full artifact bundle.

### Step 8 — Multi-symbol batch (time-box)

Thin loop: ingest/features/screen for SPY, QQQ, IWM; summary table of best model QLIKE + top factor.  
Skip if schedule slips — mark as stretch; architecture should not block it.

### Step 9 — Docs + plan status

Update `docs/research_methodology.md` (trees, SHAP, VIX, regimes, median importance).  
Package READMEs. Mark M5 DONE in `plan.md` when exit criteria met.

---



## Suggested Command Sequence

```powershell
$env:PYTHONPATH = "src"
py -m pip install -e ".[dev]"
py -m pip install -e ".[nonlinear]"   # optional but recommended for SHAP/LGBM
py -m pytest tests/unit/test_tree_models.py -q
py -m pytest tests/unit/test_importance_aggregation.py -q
py -m pytest tests/unit/test_cross_asset.py tests/unit/test_cross_asset_leakage.py -q
py -m pytest tests/unit/test_regimes.py -q
py -m pytest tests/unit/test_shap_importance.py -q
vip ingest --symbol VIX
vip features --symbol SPY --with-vix
vip screen --symbol SPY --with-vix
py -m pytest -q
```

---



## Common Pitfalls

- Fitting trees on the full sample once, then “evaluating” with walk-forward metrics (leakage).
- Computing SHAP on train and reporting it as OOS importance.
- `ffill` on VIX that accidentally uses a future print after a holiday mis-alignment — prefer `merge_asof(..., direction="backward")` on sorted indexes.
- Treating VIX as exogenous without noting SPY↔VIX contemporaneous correlation (associative, not causal).
- Letting one QLIKE permutation spike rewrite the entire factor narrative (use median).
- Making `lightgbm`/`shap` hard dependencies of the base package.
- Regime windows with zero test rows crashing the report.
- Claiming RF “wins” without showing HAR/Ridge on the **same** folds.

---



## Decisions Locked for This Walkthrough

1. RF is the required nonlinear model; LightGBM is optional.
2. Permutation screening stays on **Ridge**; trees use **SHAP** for attribution.
3. Importance ranking aggregate defaults to **median** ΔQLIKE (mean still reported).
4. Cross-asset MVP = VIX level + 1d change only.
5. Regimes = `covid_crash` + `bear_2022` + `full_sample`.
6. Report must include “What works when”.
7. No second walk-forward engine.

---



## Milestone 5 Exit Checklist

- [x] Tree models + registry + tests
- [x] Median/capped importance aggregation + tests
- [x] VIX cross-asset features + leakage tests
- [x] Regime-sliced metrics in artifacts + HTML
- [x] SHAP path (optional dep) + plot
- [x] `vip screen` robustness flags work on SPY
- [x] Methodology + READMEs updated; `plan.md` M5 DONE
- [x] Full pytest green
- [x] (Stretch) multi-symbol batch

---



## What Comes Next (M6 preview)

Platform polish: full CLI story, integration/golden tests, methodology/architecture docs, optional thin FastAPI demo endpoint.