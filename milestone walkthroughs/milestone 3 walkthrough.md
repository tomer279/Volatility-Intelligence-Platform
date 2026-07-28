# Milestone 3 Walkthrough — Baselines & Evaluation

## Objective

Evaluate simple, well-understood volatility baselines on the Milestone 2 feature matrix using walk-forward validation, and show that a HAR-style model beats a naive historical-mean forecast on held-out folds under QLIKE (and report MSE/MAE).

This milestone should prove:

- Models implement a common `VolatilityModel`-style fit/predict contract.

- Metrics suitable for volatility forecasting are implemented and tested.

- Walk-forward splitting respects time order and uses an embargo related to the forecast horizon.

- Baselines can be compared in a clear table (CLI and/or notebook).

- Results are persisted as experiment artifacts for reproducibility.

---

## Scope

### In scope

- `vip.modeling` package:

  - historical mean baseline

  - EWMA baseline

  - HAR-RV OLS baseline (using HAR feature columns already in the matrix)

- `vip.evaluation` package:

  - QLIKE, MSE, MAE

  - walk-forward splitter (expanding or rolling train window)

  - embargo / purge aligned to target horizon

  - model comparison table builder

- Application use-case: run baseline experiment on persisted features.

- Persist metrics / fold summaries under `data/artifacts/`.

- CLI: `vip evaluate` (or `vip baselines`).

- Unit tests on synthetic matrices (no network).

- Optional notebook: `notebooks/03_model_diagnostics.ipynb` for visual inspection.

### Out of scope

- Regularized linear models / factor screening (Milestone 4).

- Tree models / SHAP (Milestone 5).

- HTML research reports (Milestone 4+).

- Hyperparameter search grids (keep defaults simple).

- Live trading or position sizing.

---

## Acceptance Criteria

Milestone 3 is complete when all of the following are true:

1. Three baselines are implemented and unit-tested: historical mean, EWMA, HAR-RV OLS.

2. Metrics QLIKE, MSE, MAE are implemented and unit-tested.

3. Walk-forward evaluation runs without leakage (embargo ≥ horizon).

4. `vip evaluate --symbol SPY` (or agreed command) prints a comparison table.

5. On SPY held-out aggregation, HAR-RV OLS beats historical mean on **primary metric QLIKE** (lower is better).

6. Fold/metrics artifacts are written under `data/artifacts/`.

7. Full pytest suite remains green (no network in unit tests).

---

## Locked Research Defaults

| Setting | Value |
|---------|--------|
| Symbol | SPY |
| Input | `data/processed/SPY/features.parquet` |
| Target column | `target_rv_cc_5d` |
| HAR feature columns | `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d` |
| Primary metric | QLIKE (lower is better) |
| Secondary metrics | MSE, MAE |
| Horizon / embargo | 5 trading days (embargo ≥ 5) |

---

## Target Folder Additions

```text
src/vip/

  modeling/

    __init__.py

    [README.md](http://README.md)

    [base.py](http://base.py)                 # shared helpers / typing aliases if needed

    [baselines.py](http://baselines.py)            # HistoricalMean, EWMA, HarRvOls

    [registry.py](http://registry.py)             # optional model registry

  evaluation/

    __init__.py

    [README.md](http://README.md)

    [metrics.py](http://metrics.py)              # qlike, mse, mae

    [splitting.py](http://splitting.py)            # walk-forward fold generator

    walk_[forward.py](http://forward.py)         # run models across folds

    [comparison.py](http://comparison.py)           # summary table

  application/

    run_baseline_[experiment.py](http://experiment.py)

  cli/

    commands/

      [evaluate.py](http://evaluate.py)

configs/

  experiments/

    baselines_spy.yaml      # optional but recommended

tests/

  unit/

    test_[metrics.py](http://metrics.py)

    test_[baselines.py](http://baselines.py)

    test_[splitting.py](http://splitting.py)

    test_walk_[forward.py](http://forward.py)

    test_run_baseline_[experiment.py](http://experiment.py)

notebooks/

  03_model_diagnostics.ipynb   # optional
```

Optional dependency additions in `pyproject.toml`:

- `statsmodels` (HAR OLS with a clear econometric story)
- or pure `numpy.linalg.lstsq` / tiny OLS helper to avoid a new dependency

**Recommendation for portfolio clarity:** use `statsmodels` for HAR-RV OLS; keep historical mean / EWMA in pure pandas/numpy.

---

## Research Contract

### Prediction task

At each session `t` in a test fold:

- Features `X_t` are known at `t` (already enforced in M2).
- Target `y_t = target_rv_cc_5d` is the forward 5d RV.
- Model must be fit **only** on training rows strictly before the test block (after embargo).

### Walk-forward design (default)

Use an **expanding** training window:

1. Choose `n_folds` (start with 5) or fixed test length (e.g. 126 trading days ≈ 6 months).
2. For fold `k`:
  - `test` = next contiguous block of sessions
  - `embargo` = `horizon_days` sessions immediately before test start (excluded from train)
  - `train` = all sessions before embargo start
3. Fit on train; predict on test; score metrics on test.

**Embargo rationale:** target at time near the train/test boundary overlaps future returns that enter the test period. Embargo ≥ horizon reduces label overlap leakage.

### Metrics

Let `y` = realized vol, `yhat` = forecast (both positive).

- **MSE:** `mean((y - yhat)^2)`
- **MAE:** `mean(|y - yhat|)`
- **QLIKE:** `mean(log(yhat^2) + y^2 / yhat^2)`  
  (standard vol-forecast form; lower is better)

Implementation notes:

- Clip / guard against non-positive predictions `yhat <= 0`) with a clear policy:
  - Recommendation: enforce `yhat = max(yhat, epsilon)` with small `epsilon` (e.g. `1e-8`) and document it.
- Align predictions and targets on the same index; drop NaNs before scoring.

### Baselines

1. **HistoricalMeanModel**
  - Fit: store training-target mean.
  - Predict: constant mean for all test rows.
  - Features ignored (still accept `X` for protocol uniformity).
2. **EwmaModel**
  - Fit: estimate EWMA of training target (or recursively from training series).
  - Predict: for test dates, either
    - (A) freeze last training EWMA level as constant forecast, or
    - (B) update EWMA through test using **only past realized y** (careful: using true `y` in test is leaking label into the recursive state).
  - **Recommendation for M3:** use (A) frozen end-of-train EWMA forecast (simple, no test-label leakage). Document clearly.
  - Default decay: `lambda=0.94` (RiskMetrics-style) or span config.
3. **HarRvOlsModel**
  - Fit: OLS of `y` on `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d` (+ intercept).
  - Predict: linear combination on test features.
  - This is the classic HAR-RV idea using trailing RV components already in the matrix.

Success narrative:

> HAR-RV OLS achieves lower average QLIKE than Historical Mean on walk-forward folds for SPY.

---

## Design Rules for Milestone 3

1. No peeking: fit only on train indices; never use test `y` inside `fit`.
2. Embargo ≥ `horizon_days`.
3. Models follow a shared interface `fit` / `predict`) compatible with domain protocol intent.
4. Metrics are pure functions: `(y_true, y_pred) -> float`.
5. Evaluation orchestration lives in `vip.evaluation` / application; CLI stays thin.
6. Persist enough artifacts to reproduce a table: config snapshot, fold metrics, aggregate metrics.
7. NumPy-style docstrings; avoid broad `except`; keep functions small.
8. Unit tests use synthetic feature matrices, not live market downloads.

---

## Step-by-Step Build Plan

## Step 1 — Package skeletons + dependencies

Create:

- `src/vip/modeling/` `__init__.py`, `README.md`)
- `src/vip/evaluation/` `__init__.py`, `README.md`)

Add dependency if chosen:

```toml

"statsmodels>=0.14",

```

Then:

```powershell

py -m pip install -e ".[dev]"

```

Checkpoint:

- imports succeed

---

## Step 2 — Metrics first `evaluation/metrics.py`)

Implement:

- `mse(y_true, y_pred) -> float`
- `mae(y_true, y_pred) -> float`
- `qlike(y_true, y_pred, epsilon=1e-8) -> float`

Tests `test_metrics.py`):

- known vector hand-checks
- QLIKE penalizes bad vol forecasts on a toy example
- non-positive prediction handling

Checkpoint:

- metrics tests green

---

## Step 3 — Baselines `modeling/baselines.py`)

Implement the three models with:

- `fit(features: DataFrame, target: Series) -> self`
- `predict(features: DataFrame) -> Series` (index-aligned)

HAR model should validate required columns exist.

Tests `test_baselines.py`):

- historical mean predicts training mean
- EWMA frozen predict is constant
- HAR OLS recovers coefficients on a noiseless synthetic design (or at least fits/predicts shapes)

Checkpoint:

- baseline tests green

---

## Step 4 — Walk-forward splitting `evaluation/splitting.py`)

Implement a fold generator yielding objects like:

- `train_index`, `test_index`, `fold_id`

Parameters (keep ≤5 where possible; nest config if needed):

- `n_splits` or `test_size`
- `embargo_size`
- expanding vs rolling (start with expanding only)

Tests `test_splitting.py`):

- folds are contiguous and ordered
- train max index < test min index
- embargo gap length respected
- no overlapping train/test indices

Checkpoint:

- splitting tests green

---

## Step 5 — Walk-forward runner `evaluation/walk_forward.py`)

Implement:

- For each model and fold: fit → predict → score metrics
- Return per-fold records + aggregate mean metrics

Optional helper in `comparison.py`:

- pivot into a table sorted by QLIKE ascending

Tests `test_walk_forward.py`):

- synthetic data where HAR should beat mean (construct `y` as linear function of HAR cols + noise)
- ensures models are refit per fold (e.g. spy via fake model recording train size)

Checkpoint:

- walk-forward tests green

---

## Step 6 — Application use-case

Create `run_baseline_experiment.py`:

1. Load feature matrix from `ParquetFeatureMatrixStore`.
2. Split X / y `target_rv_cc_5d`).
3. Run walk-forward for the three baselines.
4. Write artifacts via `FilesystemArtifactStore`:
  - `metrics.json` (aggregate)
  - `folds.json` (per-fold)
  - optional `comparison.json`
5. Return a result object with table-friendly summary.

Experiment id suggestion:

- `baselines-spy-{YYYYMMDD}` or hash of config

Checkpoint:

- use-case unit test with tmp_path green

---

## Step 7 — CLI `vip evaluate`

Add `commands/evaluate.py`:

- `--symbol` (default config)
- `--n-splits` (default 5)
- `--embargo` (default 5)
- prints comparison table to stdout

Example output:

```text
Baseline walk-forward results (SPY)

metric primary: qlike (lower is better)

model              qlike      mse        mae

historical_mean    [0.xxx](http://0.xxx)      [0.xxx](http://0.xxx)      [0.xxx](http://0.xxx)

ewma               [0.xxx](http://0.xxx)      [0.xxx](http://0.xxx)      [0.xxx](http://0.xxx)

har_rv_ols         [0.xxx](http://0.xxx)      [0.xxx](http://0.xxx)      [0.xxx](http://0.xxx)

artifact: data/artifacts/.../metrics.json
```

Checkpoint:

- `vip evaluate --help`

- `vip evaluate --symbol SPY` succeeds after features exist

---

## Step 8 — Docs + plan status

Update:

- `modeling/README.md`, `evaluation/README.md`

- `application/README.md`, `cli/README.md`

- `docs/README.md`

- `plan.md` Milestone 3 DONE

- optional: short methodology note in `docs/research_methodology.md` (metrics + embargo)

---

## Suggested Command Sequence

```powershell

$env:PYTHONPATH = "src"

py -m pytest tests/unit/test_[metrics.py](http://metrics.py) -q

py -m pytest tests/unit/test_[baselines.py](http://baselines.py) -q

py -m pytest tests/unit/test_[splitting.py](http://splitting.py) -q

py -m pytest tests/unit/test_walk_[forward.py](http://forward.py) -q

py -m pytest tests/unit/test_run_baseline_[experiment.py](http://experiment.py) -q
```



# data prerequisites from M1/M2
```powershell
vip features --symbol SPY

vip evaluate --symbol SPY

py -m pytest -q
```
---

## Common Pitfalls

- Fitting once on the full sample then “evaluating” by slices (not true walk-forward).

- Embargo shorter than horizon (label overlap).

- Using test realized values inside EWMA recursion.

- QLIKE with zero/negative forecasts (log/division blow-ups).

- Comparing models on different row sets/folds.

- Leaking scaling fit across folds (not needed for these baselines; avoid sklearn Pipeline fit on full data).

---

## Decisions Locked for This Walkthrough

1. Primary metric: **QLIKE** (lower better); also report MSE/MAE.

2. Baselines: **Historical Mean**, **EWMA (frozen at train end)**, **HAR-RV OLS**.

3. Walk-forward: **expanding train**, embargo = **5** trading days by default.

4. HAR features: existing `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d`.

5. Artifacts written under `data/artifacts/{experiment_id}/`.

---

## Milestone 3 Exit Checklist

- [ ] Metrics implemented + tested

- [ ] Three baselines implemented + tested

- [ ] Walk-forward splitter + runner implemented + tested

- [ ] Application use-case persists artifacts

- [ ] CLI `vip evaluate` prints comparison table

- [ ] On SPY, HAR-RV OLS beats historical mean on QLIKE

- [ ] Docs updated; Milestone 3 marked DONE in `plan.md`

- [ ] Full pytest suite green

---

## What Comes Next (Milestone 4 preview)

Regularized linear models, permutation importance, factor screening, and the first HTML research memo — using this evaluation harness as the backbone.
