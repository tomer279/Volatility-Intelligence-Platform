# `vip.application`

## Purpose

The application package contains use-cases that orchestrate domain contracts and infrastructure adapters.  

CLI commands should call these functions rather than embedding business logic.

## Modules

- `ingest_market_data.py` - Fetch, validate, and persist daily OHLCV market data.

- `__init__.py` - Public application exports.

## Key APIs

- `ingest_market_data(source, store, symbol, date_range)` - Run the ingestion pipeline.

- `IngestMarketDataResult` - Summary of a completed ingestion run `symbol`, `row_count`, dates, output path).

## Dependencies

- Depends on: domain value objects/protocols, persistence stores, ingestion adapters (via injected source).

- Must not depend on: Typer/CLI formatting details.

## Usage

Typical flow:

1. CLI loads config and builds `YFinanceMarketDataSource` + `ParquetMarketDataStore`.

2. CLI calls `ingest_market_data(...)`.

3. Use-case returns `IngestMarketDataResult` for display/logging.

## Notes

- Keep use-cases framework-agnostic and easy to unit-test with fakes.

- Prefer dependency injection of source/store over constructing them inside the use-case.