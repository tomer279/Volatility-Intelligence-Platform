# Milestone 1 Walkthrough — Data Spine

## Objective

Build the first end-to-end data ingestion path for one symbol (`SPY`) using `yfinance`, validate and normalize daily OHLCV data, and persist it as reproducible Parquet files.

This milestone should prove:

- The app can fetch market data with a concrete vendor adapter.
- The data is validated and normalized into a canonical schema.
- The persistence layer stores and retrieves the result.
- The CLI exposes ingestion through a clean command.
- Tests remain deterministic and avoid network dependency in unit tests.

---

## Scope

### In scope

- Add ingestion package structure.
- Implement `yfinance` adapter that satisfies domain protocol intent.
- Add validation and normalization logic for daily OHLCV.
- Add application use-case for ingestion orchestration.
- Add `vip ingest` CLI command.
- Add tests (unit + optional smoke integration).

### Out of scope

- Feature engineering.
- Realized volatility target computation.
- Modeling and evaluation.
- Multi-symbol batch ingestion (design-ready only, no full batch orchestration yet).

---

## Acceptance Criteria

Milestone 1 is complete when all of the following are true:

1. `vip ingest --symbol SPY` runs successfully.
2. Raw/normalized daily OHLCV for SPY is saved under configured paths.
3. Stored data can be loaded via `ParquetMarketDataStore`.
4. Validation catches obvious schema/index issues.
5. Unit tests pass without internet.
6. Existing Milestone 0 tests remain green.

---

## Target Folder Additions

```text
src/vip/
  ingestion/
    __init__.py
    README.md
    base.py                 # optional thin abstractions/helpers
    yfinance_source.py
    validators.py
    normalize.py            # optional if validators gets too large
  application/
    __init__.py
    README.md
    ingest_market_data.py
  cli/
    commands/
      __init__.py
      ingest.py
tests/
  unit/
    test_ingestion_validators.py
    test_yfinance_source.py        # mock-based, no network
    test_ingest_use_case.py
  integration/
    test_ingest_smoke.py           # optional network-marked test
```

---

## Canonical Data Contract (Daily OHLCV)

Define this once and keep it stable:

- Index:
  - `DatetimeIndex`
  - timezone-naive normalized session date (or consistently UTC-naive)
  - strictly increasing
  - unique
- Required columns:
  - `open`, `high`, `low`, `close`, `volume`
- Column rules:
  - all lowercase names
  - numeric dtypes
  - no nulls in required columns after cleaning
  - `volume >= 0`
  - `high >= max(open, close, low)` and `low <= min(open, close, high)` (where applicable)
- Metadata (if tracked later):
  - symbol, source_name, fetch timestamp

---

## Design Rules for Milestone 1

1. Keep domain clean: no `yfinance` imports in `vip.domain`.
2. `yfinance` adapter belongs only in `vip.ingestion`.
3. Validation should raise typed domain errors (`DataValidationError`).
4. Keep functions small and testable.
5. Use NumPy-style docstrings everywhere.
6. Use `py` launcher in command examples.
7. Avoid broad `except Exception`; catch specific exceptions.
8. Do not hardcode paths; use config + persistence stores.

---

## Step-by-Step Build Plan

## Step 1 — Add dependencies

Update `pyproject.toml` dependencies:

- `yfinance`
- optional now or later: `exchange-calendars`

Then install:

```powershell
py -m pip install -e ".[dev]"
```

Checkpoint:
- install succeeds
- no existing tests regress

---

## Step 2 — Create ingestion package skeleton

Create:

- `src/vip/ingestion/__init__.py`
- `src/vip/ingestion/README.md`
- `src/vip/application/__init__.py`
- `src/vip/application/README.md`
- `src/vip/cli/commands/__init__.py`

Document module responsibilities immediately (same style as Milestone 0 docs).

Checkpoint:
- imports from new packages succeed

---

## Step 3 — Implement validators first

Create `src/vip/ingestion/validators.py` with pure functions:

Suggested responsibilities:

- `validate_required_columns(frame)`
- `validate_index(frame)`
- `validate_price_volume_rules(frame)`
- `normalize_ohlcv_frame(frame)` (or put in `normalize.py`)
- `validate_and_normalize_ohlcv(frame)`

Behavior:

- normalize vendor column names (e.g. `Open` -> `open`)
- enforce canonical column set/order
- sort by index, drop duplicate index rows using explicit policy
- raise `DataValidationError` on unrecoverable issues

Checkpoint:
- unit tests for validators pass (build these before adapter)

---

## Step 4 — Implement yfinance adapter

Create `src/vip/ingestion/yfinance_source.py`.

Class idea: `YFinanceMarketDataSource`

Responsibilities:

- satisfy `MarketDataSource` protocol intent
- fetch data for `Symbol` + `DateRange`
- run validator/normalizer
- return canonical DataFrame

Notes:

- Keep fetch-specific quirks local to this module.
- Convert vendor-specific output immediately into canonical schema.
- Catch specific data-fetch exceptions and map to `DataValidationError` or `PersistenceError` as appropriate.

Checkpoint:
- mock-based unit tests pass without network

---

## Step 5 — Add ingestion use-case

Create `src/vip/application/ingest_market_data.py`.

Use-case should orchestrate:

1. load config / resolve effective symbol/date range
2. call `MarketDataSource.fetch(...)`
3. save via `ParquetMarketDataStore.save(...)`
4. return a small result object (symbol, row_count, output_path, min/max date)

Keep this as pure orchestration; no CLI formatting here.

Checkpoint:
- use-case unit tests pass with fake source/store

---

## Step 6 — Wire CLI command

Create `src/vip/cli/commands/ingest.py` and register it from `src/vip/cli/main.py`.

Command shape suggestion:

- `vip ingest --symbol SPY`
- optional flags:
  - `--start YYYY-MM-DD`
  - `--end YYYY-MM-DD`
  - `--force-refresh` (placeholder behavior is fine if clearly documented)

CLI command should:

- load config
- construct concrete source/store
- call application use-case
- print concise success summary

Checkpoint:
- `vip ingest --help` works
- `vip ingest --symbol SPY` writes Parquet

---

## Step 7 — Persistence integration checks

After CLI run:

- verify `data/raw/SPY/ohlcv.parquet` exists (or configured equivalent)
- load via `ParquetMarketDataStore.load(Symbol("SPY"))`
- print row count / head in a local manual check script or REPL

Checkpoint:
- end-to-end happy path works locally

---

## Step 8 — Testing strategy

### Unit tests (required)

- `test_ingestion_validators.py`
  - missing columns
  - unsorted index
  - duplicate index
  - invalid high/low constraints
  - successful normalization path

- `test_yfinance_source.py`
  - mock yfinance call output
  - ensure normalization invoked
  - ensure expected columns and sorted index

- `test_ingest_use_case.py`
  - fake source + fake store orchestration
  - verifies save called with expected symbol/data

### Integration test (optional but useful)

- `tests/integration/test_ingest_smoke.py`
  - network-marked test
  - fetch tiny date window
  - assert output file created and readable

Keep CI/unit tests network-free.

Checkpoint:
- `py -m pytest -q` green (excluding optional integration if not run)

---

## Step 9 — Documentation updates

Update:

- `plan.md` status section with Milestone 1 progress
- `src/vip/ingestion/README.md`
- `src/vip/application/README.md`
- root/docs index if needed

Add a short “Data Contract” section to ingestion README.

Checkpoint:
- docs reflect actual implemented behavior

---

## Suggested Command Sequence (when implementing)

```powershell
py -m pip install -e ".[dev]"
py -m pytest tests/unit/test_ingestion_validators.py -q
py -m pytest tests/unit/test_yfinance_source.py -q
py -m pytest tests/unit/test_ingest_use_case.py -q
vip ingest --help
vip ingest --symbol SPY
py -m pytest -q
```

---

## Common Pitfalls to Avoid

- Letting vendor column names leak beyond adapter.
- Mixing validation logic directly in CLI command.
- Using broad `except` blocks.
- Writing network-dependent unit tests.
- Hardcoding `data/raw` paths in command code instead of using config.
- Skipping canonical schema definition and “fixing later”.

---

## Milestone 1 Exit Checklist

- [ ] `yfinance` adapter implemented
- [ ] validators implemented with clear canonical schema
- [ ] ingestion use-case implemented
- [ ] CLI command `vip ingest` implemented
- [ ] data saved to Parquet through persistence layer
- [ ] unit tests added and passing
- [ ] docs updated

---