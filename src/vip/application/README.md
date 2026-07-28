# `vip.application`

## Purpose
The application package contains use-cases that orchestrate domain contracts and infrastructure adapters.  
CLI commands should call these functions rather than embedding business logic.

## Modules
- `ingest_market_data.py` - Fetch, validate, and persist daily OHLCV market data.
- `build_feature_matrix.py` - Load OHLCV, build features/target, and persist the matrix.
- `run_baseline_experiment.py` - Walk-forward baseline evaluation and artifact persistence.
- `__init__.py` - Public application exports.

## Key APIs
- `ingest_market_data(source, store, symbol, date_range)` - Run the ingestion pipeline.
- `IngestMarketDataResult` - Summary of a completed ingestion run.
- `build_and_persist_feature_matrix(market_store, feature_store, symbol, ...)` - Build and save features.
- `BuildFeatureMatrixResult` - Summary of a completed feature-matrix build.
- `run_baseline_experiment(feature_store, artifact_store, symbol, ...)` - Evaluate baselines and write metrics.
- `BaselineExperimentResult` - Summary table, fold metrics, and winning model.

## Dependencies
- Depends on: domain value objects/protocols, persistence stores, features pipeline, modeling baselines, evaluation walk-forward, ingestion adapters (via injected source).
- Must not depend on: Typer/CLI formatting details.

## Usage
Ingestion:
1. CLI builds `YFinanceMarketDataSource` + `ParquetMarketDataStore`.
2. CLI calls `ingest_market_data(...)`.
3. Use-case returns `IngestMarketDataResult`.

Features:
1. CLI builds raw `ParquetMarketDataStore` + processed `ParquetFeatureMatrixStore`.
2. CLI calls `build_and_persist_feature_matrix(...)`.
3. Use-case returns `BuildFeatureMatrixResult`.

Baselines:
1. CLI builds `ParquetFeatureMatrixStore` + `FilesystemArtifactStore`.
2. CLI calls `run_baseline_experiment(...)`.
3. Use-case returns `BaselineExperimentResult` and writes `metrics.json` / `folds.json`.

## Notes
- Keep use-cases framework-agnostic and easy to unit-test with fakes.
- Prefer dependency injection of source/store over constructing them inside the use-case.