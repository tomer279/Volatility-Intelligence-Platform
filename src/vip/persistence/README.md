# `vip.persistence`

## Purpose

The persistence package provides filesystem-backed storage adapters for market data and experiment artifacts.  

It abstracts read/write behavior behind simple interfaces so pipelines stay storage-agnostic.

## Modules

- `parquet_store.py` - Parquet-based market data store keyed by symbol.

- `artifact_store.py` - JSON artifact store keyed by experiment id.

- `__init__.py` - Public persistence exports.

## Key APIs

- `ParquetMarketDataStore` - Save/load normalized OHLCV tables.

- `FilesystemArtifactStore` - Save/load experiment metadata/artifacts as JSON.

- `symbol_path(symbol)` - Deterministic symbol-to-file mapping.

- `experiment_dir(experiment_id)` - Deterministic experiment directory mapping.

## Dependencies

- Depends on: `pandas`, `pyarrow`, stdlib filesystem/json tools, domain value objects/errors.

- Must not depend on: data vendor clients, modeling code, plotting/report generation code.

## Usage

Typical flow:

1. Ingestion fetches market data.

2. `ParquetMarketDataStore.save(...)` persists tables.

3. Evaluation/reporting writes metrics with `FilesystemArtifactStore.write_json(...)`.

## Notes

- Raise `PersistenceError` for missing/unreadable data.

- Keep file naming deterministic for reproducibility and easy debugging.