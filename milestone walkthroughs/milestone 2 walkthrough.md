# Milestone 2 Walkthrough — Features & Targets

## Objective

Build a reproducible feature matrix and realized-volatility target from ingested daily OHLCV (SPY), with strict temporal alignment and leakage tests.

This milestone should prove:

- Close-to-close realized volatility can be computed for a forecast horizon.
- Feature families (returns, HAR lags, range, volume) are generated from information available at time `t` only.
- Features and targets are joined into a single aligned matrix ready for modeling.
- A feature registry makes new features addable without rewriting the pipeline.
- Leakage is tested, not only documented.

---

## Scope

### In scope

- `vip.features` package (targets, feature builders, registry, pipeline).
- Primary target: next-5-trading-day close-to-close realized volatility.
- Feature families:
  - returns (lags)
  - HAR-style realized-vol lags (daily / weekly / monthly style windows)
  - range (high-low based)
  - volume (level / change)
- Application use-case: build feature matrix from stored OHLCV.
- Persist feature matrix (Parquet under `data/processed/`).
- CLI: `vip features` (or `vip build-features`).
- Unit tests with synthetic OHLCV (no network).
- Leakage / alignment tests.
- Column dictionary documented in `features/README.md`.

### Out of scope

- Modeling / walk-forward evaluation (Milestone 3).
- Cross-asset features (VIX) (Milestone 5).
- Parkinson / Garman–Klass estimators as primary target (optional later).
- Multi-symbol batch feature builds.
- Visualization / HTML reports.

---

## Acceptance Criteria

Milestone 2 is complete when all of the following are true:

1. Given ingested SPY OHLCV, a feature matrix Parquet is written under `data/processed/`.
2. Target column exists for 5-day close-to-close RV (name documented).
3. Feature columns are documented in a column dictionary.
4. Leakage tests pass (features at `t` do not use future prices beyond `t`; target uses `(t, t+h]`).
5. `vip features --symbol SPY` (or agreed command) succeeds end-to-end.
6. Unit tests pass without internet; Milestone 0/1 tests remain green.

---

## Locked Research Defaults (from plan)

| Setting | Value |
|---------|--------|
| Symbol (flagship) | SPY |
| Primary horizon | 5 trading days |
| Primary target | close-to-close realized volatility |
| Primary metric (later M3) | QLIKE |

---

## Target Folder Additions

```text
src/vip/
  features/
    __init__.py
    README.md                 # includes column dictionary
    targets.py                # RV estimators + forward horizons
    returns.py
    realized.py               # trailing RV used as features / HAR inputs
    har.py
    range_features.py
    volume_features.py
    registry.py
    pipeline.py
    leakage.py                # optional helpers used by tests/pipeline
  application/
    build_feature_matrix.py
  cli/
    commands/
      features.py
tests/
  unit/
    test_feature_targets.py
    test_feature_builders.py
    test_feature_registry.py
    test_feature_pipeline.py
    test_feature_leakage.py
    test_build_feature_matrix_use_case.py
```

---

## Canonical Research Contract

### Time index

- Same session dates as canonical OHLCV (tz-naive, normalized, unique, sorted).
- One row = one trading session `t` (end-of-day information set).

### Target (label) at row `t`

For horizon `h = 5` trading days:

1. Compute daily log returns: `r_i = log(close_i / close_{i-1})`.
2. Future realized variance over the next `h` sessions:
   `RV2_{t,h} = sum_{i=1..h} r_{t+i}^2`
3. Realized volatility:
   `RV_{t,h} = sqrt(RV2_{t,h})`
4. Optional display annualization (decide once and document):
   - Research default recommendation: **store non-annualized RV** in the matrix; annualize only in reports later.
   - Column name suggestion: `target_rv_cc_5d`

**Critical:** target at `t` uses returns **after** `t` (`t+1 ... t+h`). The last `h` rows of the sample will have missing targets and must be dropped (or left NaN and filtered before modeling).

### Features at row `t`

May use only prices/volumes with timestamp `<= t` (including close of day `t`).

Suggested initial columns (names can be adjusted, but document them):

| Column | Family | Definition (info ≤ t) |
|--------|--------|------------------------|
| `ret_1d` | returns | `log(close_t / close_{t-1})` |
| `ret_5d` | returns | `log(close_t / close_{t-5})` |
| `rv_cc_1d` | realized | `|ret_1d|` or `sqrt(ret_1d^2)` |
| `rv_cc_5d` | HAR | sqrt of sum of last 5 squared daily returns ending at `t` |
| `rv_cc_21d` | HAR | same with 21-day window ending at `t` |
| `range_1d` | range | `(high_t - low_t) / close_t` |
| `range_5d_mean` | range | mean of `range_1d` over last 5 days ending at `t` |
| `volume_z_21d` | volume | `(volume_t - mean_21) / std_21` using windows ending at `t` |

HAR-style features use **trailing** (past) realized vol, not the forward target.

### Output matrix schema

- Index: session date
- Columns: feature columns + `target_rv_cc_5d`
- Rows with NaN target or NaN required features: drop before save (or save full and drop in use-case — pick one policy and document it)
- Recommendation: **drop incomplete rows** in the build use-case so modeling always loads a clean matrix.

---

## Design Rules for Milestone 2

1. No leakage: features ≤ `t`; target uses `t+1..t+h`.
2. Feature builders are pure functions/classes over OHLCV DataFrames.
3. Registry maps string names → builders (config-driven later).
4. Domain stays free of feature math if possible; keep formulas in `vip.features`.
5. Raise `DataValidationError` / `LeakageError` for contract violations.
6. NumPy-style docstrings; prefer ≤5 parameters; avoid broad `except`.
7. Tests use synthetic OHLCV fixtures, not live Yahoo downloads.
8. Do not import sklearn yet (unless a tiny transform helper is truly needed — prefer pandas/numpy only).

---

## Step-by-Step Build Plan

## Step 1 — Package skeleton + README contract

Create:

- `src/vip/features/__init__.py`
- `src/vip/features/README.md` (purpose + data contract + empty column dictionary table)

Checkpoint:

- import `vip.features` succeeds

---

## Step 2 — Targets first (`targets.py`)

Implement:

- `daily_log_returns(close) -> Series`
- `realized_variance_forward(returns, horizon) -> Series`  # aligned to `t`, using future window
- `realized_volatility_forward(returns, horizon) -> Series`
- `build_target_rv_cc(ohlcv, horizon_days=5) -> Series` named `target_rv_cc_5d`

Policy:

- Use trading-day shifts via the DataFrame index order (assume sessions already filtered to trading days by OHLCV).
- Do not use calendar-day rolling for the primary target.

Tests (`test_feature_targets.py`):

- Known toy prices → expected RV hand-calculated
- Last `horizon` values are NaN
- Horizon < 1 raises

Checkpoint:

- target unit tests green

---

## Step 3 — Trailing realized helpers (`realized.py`)

Implement trailing (backward) RV used by HAR/features:

- `realized_volatility_trailing(returns, window) -> Series`  # ends at `t`

Tests:

- Window uses only past/current returns
- Compare against manual sum of squared returns

---

## Step 4 — Feature family modules

Implement small builders (each returns a DataFrame or Series with stable column names):

- `returns.py` → `ret_1d`, `ret_5d`
- `har.py` → `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d` (trailing)
- `range_features.py` → `range_1d`, `range_5d_mean`
- `volume_features.py` → `volume_z_21d`

Keep each module focused; share return computation rather than duplicating.

Checkpoint:

- `test_feature_builders.py` green

---

## Step 5 — Registry (`registry.py`)

Implement:

- `FeatureSpec(name, builder, description)`
- `FeatureRegistry.register(...)`
- `FeatureRegistry.build_all(ohlcv, names: list[str] | None) -> DataFrame`

Default registered set = Milestone 2 families above.

Checkpoint:

- `test_feature_registry.py` green

---

## Step 6 — Pipeline (`pipeline.py`)

Implement:

- `build_feature_matrix(ohlcv, horizon_days=5, feature_names=None) -> DataFrame`
  1. validate OHLCV via existing `validate_and_normalize_ohlcv` (or assume already canonical and re-validate lightly)
  2. build features via registry
  3. build target
  4. concatenate on index
  5. drop rows with NaNs in features/target
  6. return matrix

Checkpoint:

- `test_feature_pipeline.py` green on synthetic data

---

## Step 7 — Leakage tests (`test_feature_leakage.py`) — non-negotiable

Minimum assertions:

1. **Target shift:** mutating `close` only at `t+1` changes target at `t`, not features at `t` (or carefully designed probes).
2. **Feature causality:** for each feature column at date `t`, recomputing features on a frame truncated after `t` yields the same feature values at `t`.
3. **No peeking:** features at last available date do not require future rows.

Also assert:

- `LeakageError` is available and used if you add an explicit checker helper.

Checkpoint:

- leakage tests green

---

## Step 8 — Application use-case

Create `build_feature_matrix.py`:

1. Load OHLCV from `ParquetMarketDataStore` (raw or processed — recommend **raw** as source of truth for M2).
2. Run `build_feature_matrix(...)`.
3. Save to processed store path, e.g. `data/processed/{SYMBOL}/features.parquet`.
4. Return summary: symbol, rows, n_features, path, date span.

Extend persistence if needed:

- Either reuse Parquet store with a different root (`processed_dir`) and filename (`features.parquet`), or add a small `FeatureMatrixStore`.
- Recommendation: keep `ParquetMarketDataStore`-style helper or generalize filename; avoid over-engineering.

Checkpoint:

- `test_build_feature_matrix_use_case.py` with tmp_path green

---

## Step 9 — CLI command

Add `vip features`:

- `--symbol` (default config)
- optional `--horizon` (default 5)
- reads config paths
- prints row count, feature count, output path

Dates are strings if needed (Typer lesson from M1).

Checkpoint:

- `vip features --help`
- `vip features --symbol SPY` after ingest

---

## Step 10 — Docs + column dictionary

Update:

- `features/README.md` with full column dictionary
- `application/README.md` with new use-case
- `cli/README.md` with `features` command
- `docs/README.md` package index
- `plan.md` Milestone 2 status when done

---

## Suggested Command Sequence

```powershell
# after coding each step
$env:PYTHONPATH = "src"
py -m pytest tests/unit/test_feature_targets.py -q
py -m pytest tests/unit/test_feature_builders.py -q
py -m pytest tests/unit/test_feature_registry.py -q
py -m pytest tests/unit/test_feature_pipeline.py -q
py -m pytest tests/unit/test_feature_leakage.py -q
py -m pytest tests/unit/test_build_feature_matrix_use_case.py -q

vip features --help
vip ingest --symbol SPY --start 2018-01-01 --end 2024-12-31
vip features --symbol SPY
py -m pytest -q
```

---

## Common Pitfalls

- Using `rolling(5)` on calendar days when index has gaps — prefer index-position windows on trading-day series.
- Building target with inclusive wrong window (`t` to `t+h-1` vs `t+1` to `t+h`).
- Letting `Adj Close` sneak into returns (use canonical `close`).
- Fitting scalers on full sample (not needed in M2; avoid sklearn StandardScaler on full data).
- Saving matrices that still contain NaNs.
- Network tests for features (unnecessary).

---

## Decisions Locked for This Walkthrough

1. Primary target: **forward 5d close-to-close RV**, column `target_rv_cc_5d`.
2. Store **non-annualized** RV in the matrix.
3. Drop rows with any NaN in features/target before save.
4. Source OHLCV from **raw** Parquet produced by Milestone 1.
5. Output path: `data/processed/{SYMBOL}/features.parquet`.

---

## Milestone 2 Exit Checklist

- [ ] `vip.features` package implemented
- [ ] Target + feature families implemented
- [ ] Registry + pipeline implemented
- [ ] Leakage tests green
- [ ] Use-case + CLI `vip features` working
- [ ] SPY feature matrix Parquet written
- [ ] Column dictionary documented
- [ ] Full pytest suite green
- [ ] Docs updated; Milestone 2 marked DONE in `plan.md`

---

## What Comes Next (Milestone 3 preview)

Baselines (HAR-RV OLS, EWMA, historical mean), QLIKE/MSE/MAE, walk-forward with embargo — consuming this feature matrix.

--