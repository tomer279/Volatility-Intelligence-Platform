# `vip.ingestion`

## Purpose

This package fetches external market data and converts it into the platform's canonical daily OHLCV schema.

## Modules

- `validators.py` - Validation and normalization for canonical OHLCV.

- `yfinance_source.py` - Yahoo Finance adapter implementation.

- `__init__.py` - Public ingestion exports.

## Key APIs

- `validate_and_normalize_ohlcv(frame)` - Normalize and validate OHLCV tables.

- `YFinanceMarketDataSource.fetch(symbol, date_range)` - Fetch and return canonical OHLCV.

- `YFinanceMarketDataSource.source_name()` - Source identifier (`yfinance`).

## Data contract

Canonical output must have:

- columns: `open`, `high`, `low`, `close`, `volume`

- UTC-normalized tz-naive `DatetimeIndex`

- sorted ascending, unique index

- valid OHLC bounds and non-negative volume