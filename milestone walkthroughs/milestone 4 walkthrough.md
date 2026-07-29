# Milestone 4 Walkthrough — Factor Intelligence

## Objective

Move from “which baseline forecasts RV best?” to “which factors matter?”, using regularized linear models on the full M2 feature set, walk-forward permutation importance, a factor-screening use-case, and a first HTML research memo.

This milestone should prove:

- Ridge / Lasso / ElasticNet implement the same `fit` / `predict` contract as M3 baselines.
- Scaling and model fit happen **per fold** on train only (no leakage).
- Permutation importance ranks predictors under the research loss (QLIKE primary; document secondary).
- Factor rankings are summarized with **stability across folds** and explicit correlation caveats.
- An HTML report under `data/artifacts/` is skimmable by a PM.

---

## Scope

### In scope

- `vip.modeling.regularization` — Ridge, Lasso, ElasticNet wrappers (`scikit-learn`)
- Shared helper: train-only `StandardScaler` + model (pipeline-style adapter)
- Wire models into existing `run_walk_forward` (reuse M3 harness)
- `vip.evaluation.importance` — permutation importance (per fold + aggregate)
- `vip.evaluation.stability` — fold-to-fold rank/score stability helpers
- Application: `screen_factors.py` (+ optional thin `generate_report.py`)
- `vip.visualization` — minimal importance / stability plots (matplotlib)
- `vip.reporting` — Jinja2 HTML memo
- Persist JSON + figures + `report.html` under `data/artifacts/`
- CLI: `vip screen` (and/or `vip report`)
- Unit tests on synthetic matrices (no network)
- Deps: `scikit-learn`, `matplotlib`, `jinja2`

### Out of scope

- LightGBM / RF / SHAP (M5)
- VIX / cross-asset features (M5)
- Multi-symbol batch screening (design OK; implement M5)
- Diebold–Mariano / formal inference
- Hyperparameter grids beyond a small fixed default set
- Markdown report polish (optional stub only)

---

## Acceptance Criteria

1. Ridge, Lasso, ElasticNet implemented + unit-tested; positive prediction floor (same policy as M3).
2. Walk-forward evaluation of regularized models vs HAR-RV OLS on SPY (same folds/embargo as M3).
3. Permutation importance computed fold-wise on held-out test rows; aggregated mean ± fold stability.
4. `vip screen --symbol SPY` produces a ranked factor table and writes artifacts.
5. HTML report includes: research question, locked defaults, model horse-race (QLIKE), ranked factors, caveats (correlation / multicollinearity / regime).
6. Leakage tests: scaler/model never fit on test; importance permutations only shuffle feature columns in the evaluation window.
7. Full pytest suite green (no network in unit tests).

---

## Locked Research Defaults

| Setting | Value |
|---------|--------|
| Symbol | SPY |
| Input | `data/processed/SPY/features.parquet` |
| Target | `target_rv_cc_5d` |
| Predictor set | all M2 feature cols (exclude target) |
| Primary metric | QLIKE (lower better) |
| Secondary | MSE, MAE |
| Walk-forward | expanding, `n_splits=5`, `embargo=5` |
| Importance metric | QLIKE (negate or use as scoring so “higher importance = more damage when shuffled”) |
| Regularization defaults | Ridge `alpha=1.0`; Lasso `alpha=0.001`; ElasticNet `alpha=0.001`, `l1_ratio=0.5` (tune only if needed) |
| Scaling | `StandardScaler` fit on train fold only |
| HAR baseline | keep `HarRvOlsModel` as the “simple structure” reference |

**Predictor columns (M2 dictionary):**  
`ret_1d`, `ret_5d`, `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d`, `range_1d`, `range_5d_mean`, `volume_z_21d`

---

## Target Folder Additions

```text
src/vip/
  modeling/
    regularization.py      # Ridge/Lasso/ElasticNet + ScaledLinearModel helper
    registry.py            # fill empty registry: name → factory
  evaluation/
    importance.py          # permutation importance over folds
    stability.py           # rank stability / mean importance tables
  visualization/
    __init__.py
    README.md
    styles.py
    importance_plots.py    # bar chart of mean importance
    # optional: stability heatmap later
  reporting/
    __init__.py
    README.md
    templates/
      factor_screen.html.j2
    html_report.py
    experiment_summary.py  # small dataclass → template context
  application/
    screen_factors.py
    generate_report.py     # optional thin wrapper
  cli/commands/
    screen.py              # vip screen
    # optional: report.py

configs/experiments/
  factor_screen_spy.yaml   # optional but recommended

tests/unit/
  test_regularization.py
  test_importance.py
  test_stability.py
  test_screen_factors.py
  test_html_report.py
  test_importance_leakage.py   # scaler/fit isolation
```

---

## Research Contract

### Models

Each regularized model:

1. On `fit(X_train, y_train)`: drop NaN rows jointly; fit scaler on `X_train`; fit sklearn estimator on scaled `X`.
2. On `predict(X)`: transform with **train** scaler; predict; clip `yhat = max(yhat, 1e-8)`.
3. Expose `feature_names_` after fit (for importance).

Prefer one internal class (e.g. `ScaledLinearModel`) parameterized by estimator factory, then thin public wrappers — keeps ≤5 public params and avoids three near-copies.

### Permutation importance

For each fold:

1. Fit model on train (after embargo).
2. Baseline score = QLIKE on test.
3. For each feature `j`, shuffle column `j` in the **test** frame only (seeded), rescore QLIKE.
4. Importance_j = QLIKE_permuted − QLIKE_baseline (higher = more harmful when destroyed = more important).

Aggregate across folds: mean importance, std, and rank frequency (how often feature is in top-k).

**Do not** use sklearn’s default `r2` scorer — wrap QLIKE explicitly.

### Factor screening narrative

Success story (portfolio demo):

> On SPY next-5d RV, HAR-style trailing RV features dominate mean permutation importance under a Ridge/Lasso walk-forward; range/volume add secondary signal; rankings are unstable for weak factors — treat collinear HAR lags as a family, not independent discoveries.

### HTML report (minimum sections)

1. Title + symbol + date + experiment id  
2. Locked methodology (target, metric, embargo)  
3. Model comparison table (include HAR OLS + at least one regularized model)  
4. Ranked factor table (mean importance, stability)  
5. One importance bar figure (embedded base64 or file link)  
6. Caveats (correlation among HAR lags; no causal claim; sample/regime limits)

---

## Design Rules

1. Reuse `run_walk_forward` / `generate_expanding_folds` — do not fork a second CV engine.
2. Fit scaler + model only on train indices each fold.
3. Importance shuffles test features only; never refit inside a permutation.
4. CLI thin; orchestration in application; math in evaluation/modeling.
5. NumPy docstrings; module/class docs per your rules; ≤5 params (bundle options in a small frozen config dataclass if needed).
6. No broad `except`; typed `DataValidationError` / `PersistenceError`.
7. Extend `FilesystemArtifactStore` with `write_text` / `write_bytes` (or write HTML via Path in reporting after `experiment_dir`) — keep JSON protocol intact.
8. `scikit-learn` is an infrastructure adapter behind `VolatilityModel`-shaped classes.

---

## Step-by-Step Build Plan

### Step 1 — Dependencies + package skeletons

Add to `pyproject.toml`:

```toml
"scikit-learn>=1.4",
"matplotlib>=3.8",
"jinja2>=3.1",
```

Create empty packages: `visualization/`, `reporting/` (+ READMEs).  
`py -m pip install -e ".[dev]"`  
**Checkpoint:** imports succeed.

### Step 2 — Regularized models (`modeling/regularization.py`)

Implement Ridge / Lasso / ElasticNet with shared scaling.  
Tests: shapes; noiseless recovery for Ridge; Lasso sparsity on sparse synthetic design; prediction floor; unfitted predict raises.  
**Checkpoint:** `test_regularization.py` green.

### Step 3 — Registry + walk-forward smoke

Fill `registry.py`; run walk-forward on synthetic data where `y` depends on a subset of columns — Lasso/Ridge should beat historical mean on QLIKE.  
**Checkpoint:** regularized models plug into `run_walk_forward` unchanged.

### Step 4 — Permutation importance (`evaluation/importance.py`)

API sketch (≤5 params — nest the rest):

```python
def permutation_importance_folds(
    features, target, model_factory, fold_spec, n_repeats=5
) -> pd.DataFrame
```

`model_factory` = zero-arg callable returning a fresh model (refit per fold).  
Tests: known signal column ranks above pure noise; shuffle doesn’t mutate original frame.  
**Checkpoint:** `test_importance.py` green.

### Step 5 — Stability helpers (`evaluation/stability.py`)

Mean/std importance; top-k hit rate; optional Spearman rank correlation across adjacent folds.  
**Checkpoint:** table builder unit-tested.

### Step 6 — Screen use-case (`application/screen_factors.py`)

1. Load features (same split as baselines).  
2. Walk-forward horse-race: `har_rv_ols` + `ridge` (+ optional lasso/elasticnet).  
3. Permutation importance for primary screening model (recommend **Ridge** for dense stable ranks; report Lasso nonzero set as secondary).  
4. Persist `metrics.json`, `importance.json`, `factor_ranking.json`.  
**Checkpoint:** `test_screen_factors.py` with `tmp_path` green.

### Step 7 — Visualization

Bar chart of mean permutation importance → PNG under experiment dir.  
Keep styling minimal (`styles.py` with a few rcParams).  
**Checkpoint:** PNG written in use-case or report step.

### Step 8 — HTML report (`reporting/`)

Jinja2 template + `render_factor_screen_report(context) -> str`.  
Embed table HTML + image.  
Extend artifact store or write `report.html` beside JSON.  
**Checkpoint:** `test_html_report.py` asserts key headings present.

### Step 9 — CLI `vip screen`

```text
vip screen --symbol SPY --n-splits 5 --embargo 5
```

Print ranked factors + path to `report.html`.  
Optional: `vip report --experiment-id ...` later; M4 can render report inside `screen`.

### Step 10 — Docs + plan status

Update package READMEs, `docs/research_methodology.md` (regularization, importance, caveats), mark M4 DONE in `plan.md` when exit criteria met.

---

## Suggested Command Sequence

```powershell
$env:PYTHONPATH = "src"
py -m pip install -e ".[dev]"
py -m pytest tests/unit/test_regularization.py -q
py -m pytest tests/unit/test_importance.py tests/unit/test_stability.py -q
py -m pytest tests/unit/test_screen_factors.py tests/unit/test_html_report.py -q
vip features --symbol SPY
vip evaluate --symbol SPY
vip screen --symbol SPY
py -m pytest -q
```

---

## Common Pitfalls

- Fitting `StandardScaler` on the full sample once (leakage).
- Using sklearn `permutation_importance` with default R² while claiming QLIKE research.
- Interpreting Lasso zeros as “factor doesn’t matter” without checking Ridge/HAR collinearity.
- Treating `rv_cc_1d/5d/21d` as independent discoveries (they are a HAR family).
- Refitting the model inside each permutation (slow + wrong for this design).
- Importance on train instead of test (overconfident in-sample ranks).
- HTML without methodology caveats (overclaims for a portfolio piece).

---

## Decisions Locked for This Walkthrough

1. Screening model for importance: **Ridge** (primary); Lasso sparsity as supporting evidence.  
2. Importance = **ΔQLIKE** under column permutation on test folds.  
3. Always compare to **HAR-RV OLS** in the horse-race table.  
4. Report artifact: `data/artifacts/factor-screen-spy-{date}/report.html`.  
5. No SHAP / trees / VIX in M4.

---

## Milestone 4 Exit Checklist

- [x] Regularized models + tests
- [x] Permutation importance + leakage tests
- [x] Stability / ranking table
- [x] `screen_factors` persists artifacts
- [x] Importance plot + HTML report
- [x] `vip screen` works on SPY
- [x] Methodology + READMEs updated; `plan.md` M4 DONE
- [x] Full pytest green

---

## What Comes Next (M5 preview)

Nonlinear models, SHAP, VIX/cross-asset features, regime slices (COVID / 2022) — “what works when.”

---