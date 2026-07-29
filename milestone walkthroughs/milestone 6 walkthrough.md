# Milestone 6 Walkthrough — Platform Polish

## Objective

Tie the M0–M5 research platform into a reproducible, documented, testable product: a single CLI command reproduces the flagship SPY/QQQ study end-to-end, integration tests pin output metrics against golden files, and docs let a new contributor understand the architecture and extend the feature set.

This milestone should prove:

- One command (`vip run`) chains ingest → features → screen for one or more symbols, producing the full HTML report bundle without manual intermediate steps.
- A frozen integration test replays the pipeline on synthetic (or cached) data and asserts output metrics match golden-file snapshots — regressions break CI.
- Docs cover methodology (complete through M5), architecture (layer diagram + data flow), and a "how to add a feature" tutorial.
- (Optional) A thin FastAPI endpoint serves experiment artifacts as JSON for demo purposes.

---



## Scope



### In scope

- `vip.application.run_study` — composite use-case that orchestrates ingest → features → screen for N symbols
- `vip.cli.commands.run` — `vip run` command wiring the composite use-case
- Integration test suite under `tests/integration/` with golden-file metric snapshots
- Deterministic test fixtures (mock yfinance or cached Parquet) so integration tests are network-free
- Complete `docs/research_methodology.md` (add M5 content: RF, SHAP, VIX, regimes, median importance)
- `docs/architecture.md` — layer diagram, data flow, key abstractions
- `docs/how_to_add_feature.md` — step-by-step tutorial for contributors
- Optional: `src/vip/api/` — thin FastAPI app with `GET /experiments/{id}` reading from artifact store
- Optional: `fastapi` + `uvicorn` in `[api]` optional dependency group
- Update `plan.md` with M6 DONE status



### Out of scope

- Hyperparameter tuning / Optuna
- Intraday/high-frequency RV
- Options-implied surfaces
- Portfolio-of-names / cross-sectional models
- Live scheduling / production monitoring
- New models or features beyond M5

---



## Acceptance Criteria

1. `vip run --symbols SPY,QQQ --with-vix` produces HTML reports + full artifact bundles for both symbols without manual intermediate commands.
2. `vip run --symbol SPY` (single-symbol shorthand) works identically to the batch path with one symbol.
3. Integration test replays the pipeline on a frozen fixture and asserts QLIKE/MSE/MAE match golden-file values within a tolerance (e.g. `1e-6` relative).
4. Integration test is network-free (mock yfinance or pre-cached Parquet fixture).
5. `docs/research_methodology.md` covers all M1–M5 methodology: targets, features, baselines, regularized models, RF, SHAP, VIX, regimes, median importance, caveats.
6. `docs/architecture.md` documents the layer diagram (domain → config → ingestion → features → modeling → evaluation → application → cli), data flow, and extension points.
7. `docs/how_to_add_feature.md` walks a contributor through: implement function → register → leakage test → rebuild matrix.
8. (Optional) `GET /experiments/{id}` returns experiment metrics as JSON; tested with a unit test using FastAPI's `TestClient`.
9. Full pytest suite green (unit + integration); no network in tests.
10. `plan.md` M6 marked DONE with summary.

---



## Locked Research Defaults


| Setting                | Value                                                              |
| ---------------------- | ------------------------------------------------------------------ |
| Flagship symbols       | SPY, QQQ (IWM as stretch)                                          |
| Target                 | `target_rv_cc_5d`                                                  |
| Primary metric         | QLIKE (lower better)                                               |
| Secondary              | MSE, MAE                                                           |
| Walk-forward           | expanding, `n_splits=5`, `embargo=5`                               |
| Linear screening model | Ridge                                                              |
| Nonlinear default      | RandomForest (`n_estimators=200`, `max_depth=4`, `random_state=0`) |
| Cross-asset            | VIX via `--with-vix`                                               |
| Importance aggregate   | median ΔQLIKE                                                      |
| Regimes                | `covid_crash`, `bear_2022`, `full_sample`                          |
| Artifact root          | `data/artifacts/factor-screen-{symbol}-{date}/`                    |


---



## Target Folder Additions

```text
src/vip/
  application/
    run_study.py             # composite use-case: ingest → features → screen
  cli/commands/
    run.py                   # vip run
  api/                       # optional
    __init__.py
    app.py                   # FastAPI app with GET /experiments/{id}

docs/
  research_methodology.md    # complete (existing draft → full)
  architecture.md            # new
  how_to_add_feature.md      # new

tests/
  integration/
    __init__.py
    conftest.py              # shared fixtures (mock source, tmp dirs)
    test_full_pipeline.py    # golden-file integration test
  fixtures/
    golden_metrics_spy.json  # frozen expected metrics for SPY
    spy_ohlcv_fixture.parquet  # cached market data (optional)

configs/
  experiments/
    flagship_spy_qqq.yaml    # optional: frozen config for reproduction
```

---



## Research Contract



### Composite use-case (`run_study`)

For each symbol in the requested list:

1. **Ingest** — fetch OHLCV via `MarketDataSource` and persist to Parquet store. If `--skip-ingest` is set, assert data exists or raise `PersistenceError`.
2. **Ingest VIX** — if `--with-vix` is set, ingest VIX as an auxiliary symbol (same as `vip ingest --symbol VIX`).
3. **Build features** — run the feature pipeline (own-symbol + optional cross-asset). If `--skip-features` is set, assert feature matrix exists.
4. **Screen** — delegate to `screen_factors` (or `screen_batch` for multi-symbol) with the standard config.
5. Return paths to all generated reports.

The composite use-case must not duplicate logic from existing use-cases — it composes them in sequence. Each step logs progress.

### Golden-file integration test

1. Use a **deterministic fixture**: either a small cached Parquet file (50–100 rows of realistic OHLCV) or a mock `MarketDataSource` that returns fixed synthetic data with `random_state=42`.
2. Run the full pipeline: ingest → features → screen.
3. Load output `metrics.json` and assert each metric matches the golden file within relative tolerance `1e-6`.
4. If the pipeline changes in a way that legitimately alters metrics, the golden file is updated intentionally (not silently).
5. Test must complete in under 60 seconds.



### Documentation standards

- **Methodology**: written for a technically literate PM or quant analyst — explain *what* and *why*, not implementation details. Reference locked defaults. Include caveats (no causal claims, regime sensitivity, sample limitations).
- **Architecture**: written for a developer joining the project — layer diagram, dependency direction, key protocols (`MarketDataSource`, `VolatilityModel`), data flow from raw OHLCV to HTML report.
- **How to add a feature**: written as a tutorial with concrete code snippets — implement, register, test for leakage, rebuild.

---



## Design Rules

1. `run_study` composes existing use-cases (`screen_factors`, `screen_batch`, `build_feature_matrix`); no duplicated CV/importance logic.
2. Integration tests must be network-free — mock or fixture the data source.
3. Golden files are committed to the repo; updating them requires an intentional change.
4. CLI thin; orchestration in application; math stays in evaluation/modeling/features.
5. NumPy docstrings; module/class docs; ≤5 params (bundle in frozen dataclasses).
6. No broad `except`; typed domain errors.
7. FastAPI endpoint (if built) reads from `FilesystemArtifactStore` only — no new persistence layer.
8. Docs use Markdown; no auto-generated API docs in M6 (keep it manual and accurate).

---



## Step-by-Step Build Plan



### Step 1 — Composite use-case (`application/run_study.py`)

Implement `RunStudyConfig` (dataclass) and `run_study(...)` that chains ingest → features → screen for a list of symbols. Delegate to existing `build_feature_matrix`, `screen_factors`, and `screen_batch`.

For single symbol: call `screen_factors` directly.
For multiple symbols: call `run_screen_batch` (already exists from M5).

**Checkpoint:** calling `run_study` with SPY produces the same artifacts as the manual `vip ingest` + `vip features` + `vip screen` sequence.

### Step 2 — `vip run` CLI command

Register `vip run` in `main.py`:

```text
vip run --symbols SPY,QQQ --with-vix --skip-ingest --skip-features
vip run --symbol SPY                  # single-symbol shorthand
```

Flags mirror existing `screen-batch` flags plus `--skip-ingest` and `--skip-features` for skipping already-completed steps.

Print a summary table and report paths, same style as `screen-batch`.

**Checkpoint:** `vip run --symbol SPY` produces the full HTML report.

### Step 3 — Integration test fixtures

Create `tests/fixtures/` with a small deterministic dataset:

- Either a cached Parquet with ~100 rows of realistic SPY OHLCV, or
- A `MockMarketDataSource` that generates synthetic but reproducible data (`random_state=42`).

Also create a `conftest.py` under `tests/integration/` with shared fixtures (tmp dirs, stores, mock source).

**Checkpoint:** fixture loads and produces a valid feature matrix.

### Step 4 — Golden-file integration test

Write `tests/integration/test_full_pipeline.py`:

1. Run the full pipeline on the fixture data.
2. Load output `metrics.json`.
3. Compare against `tests/fixtures/golden_metrics_spy.json`.
4. Assert within tolerance.

Generate the golden file on first run, commit it, then assert against it on subsequent runs.

**Checkpoint:** `py -m pytest tests/integration/ -q` green.

### Step 5 — Complete `docs/research_methodology.md`

Add sections for:

- Tree models (RandomForest): hyperparameters, unscaled inputs, prediction floor
- SHAP attribution: fold-wise mean |SHAP| on test rows, median aggregation, not causal
- VIX cross-asset features: `vix_level`, `vix_chg_1d`, backward asof join, no forward-fill leakage
- Regime-sliced evaluation: COVID crash, 2022 bear, full sample; empty-slice handling
- Median importance: replacing mean as default to handle QLIKE spikes
- Updated caveats section

**Checkpoint:** methodology doc is self-contained for all M1–M5 decisions.

### Step 6 — Architecture doc (`docs/architecture.md`)

Document:

- Layer diagram: domain → config → ingestion → features → modeling → evaluation → visualization → reporting → application → cli
- Dependency direction (inner layers don't import outer)
- Key protocols and their implementations
- Data flow: raw OHLCV → processed features → walk-forward folds → metrics/importance → artifacts → HTML
- Extension points (registries for models and features)

**Checkpoint:** a new developer can trace a `vip screen` call from CLI to HTML output.

### Step 7 — "How to add a feature" guide (`docs/how_to_add_feature.md`)

Tutorial structure:

1. Decide which family the feature belongs to (HAR, return, range, volume, cross-asset)
2. Implement the computation function (with example)
3. Register in the feature pipeline
4. Write a leakage test (assert feature at `t` uses only data ≤ `t`)
5. Rebuild the matrix: `vip features --symbol SPY`
6. Run screen to see the new feature's importance

**Checkpoint:** doc is actionable with copy-paste code snippets.

### Step 8 — Optional: FastAPI endpoint

Add `[api]` optional dependency group (`fastapi`, `uvicorn`).

Implement `src/vip/api/app.py` with:

- `GET /experiments/{id}` — returns `metrics.json` content
- `GET /experiments/{id}/report` — returns HTML report as response
- `GET /experiments/` — lists available experiment IDs

Unit test with `TestClient`.

**Checkpoint:** `py -m pytest tests/unit/test_api.py -q` green (with `[api]` installed).

### Step 9 — Update `plan.md` + final test sweep

Mark M6 DONE with summary. Run full test suite. Verify the exit criterion:

```powershell
vip run --symbols SPY,QQQ --with-vix
```

produces complete, reproducible HTML reports.

**Checkpoint:** all tests green; plan updated.

---



## Suggested Command Sequence

```powershell
$env:PYTHONPATH = "src"
py -m pip install -e ".[dev]"
py -m pytest tests/unit/ -q                      # existing tests still green
py -m pytest tests/integration/ -q               # new integration tests
vip run --symbol SPY --with-vix                   # single-symbol end-to-end
vip run --symbols SPY,QQQ --with-vix              # flagship multi-symbol
py -m pytest -q                                   # full sweep
# optional:
py -m pip install -e ".[api]"
py -m pytest tests/unit/test_api.py -q
uvicorn vip.api.app:app --port 8000
```

---



## Common Pitfalls

- Duplicating ingest/features/screen logic in the composite use-case instead of composing existing functions.
- Integration tests that hit the network (yfinance) and become flaky — mock or cache the data source.
- Golden files that are platform-dependent (floating-point differences across OS/CPU) — use relative tolerance, not exact equality.
- Updating the pipeline without regenerating golden files — tests pass locally but fail in CI.
- Architecture doc that describes aspirational design instead of actual code — keep it grounded in current modules.
- FastAPI endpoint that reimplements artifact reading instead of using `FilesystemArtifactStore`.
- Writing a "how to add a feature" guide that skips the leakage test step.
- `vip run` that swallows errors from individual steps instead of failing fast with a clear message.

---



## Decisions Locked for This Walkthrough

1. Composite command is `vip run` (not `vip reproduce` or `vip pipeline`).
2. Integration tests use deterministic fixtures, not live network calls.
3. Golden-file tolerance: relative `1e-6` for metrics.
4. Docs are Markdown in `docs/`; no auto-generated API reference in M6.
5. FastAPI is optional — core exit criterion is CLI reproduction, not a web server.
6. No new models or features in M6 — this is a polish milestone.

---



## Milestone 6 Exit Checklist

- [x] `vip run` composite command works for single and multi-symbol
- [x] Integration test with golden-file assertions (network-free)
- [x] `docs/research_methodology.md` complete through M5
- [x] `docs/architecture.md` with layer diagram and data flow
- [x] `docs/how_to_add_feature.md` contributor tutorial
- [x] (Optional) FastAPI `GET /experiments/{id}` + unit test
- [x] `vip run --symbols SPY,QQQ --with-vix` reproduces the flagship study
- [x] Full pytest green (unit + integration)
- [x] `plan.md` M6 DONE

---



## What Comes Next (post-MVP)

Intraday/high-frequency RV, options-implied surfaces, portfolio-of-names / cross-sectional models, live scheduling, production monitoring.