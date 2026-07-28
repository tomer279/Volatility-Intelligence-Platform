# `vip.domain`

## Purpose

The domain package defines the core research language of the platform: value objects, entities, enums, protocol contracts, and typed errors.  

It should stay stable even if data vendors, model libraries, or storage technologies change.

## Modules

- `errors.py` - Platform-specific exception hierarchy.

- `enums.py` - Canonical enums for estimators, metrics, frequencies, and split modes.

- `value_objects.py` - Immutable validated value objects `Symbol`, `DateRange`, `Horizon`, `ExperimentId`).

- `entities.py` - Core domain entities for instrument, dataset references, and experiment identity.

- `protocols.py` - Structural interfaces (Protocols) for adapters `MarketDataSource`, `MarketDataStore`, `VolatilityModel`, etc.).

- `__init__.py` - Domain public exports.

## Key APIs

- `Symbol` - Normalizes and validates ticker identifiers.

- `DateRange` - Validates inclusive date windows and provides helper methods.

- `Horizon` - Represents forecast horizon in trading days.

- `ExperimentSpec` - Minimal domain representation of an experiment identity.

- `MarketDataSource` - Contract for external market data adapters.

- `VolatilityModel` - Contract for fit/predict model adapters.

## Dependencies

- Depends on: Python stdlib, typing tools, and minimal shared libs.

- Must not depend on: vendor SDKs `yfinance`, `polygon`), ML frameworks `sklearn`), or plotting/reporting libraries.

## Usage

Typical imports:

- `from vip.domain import Symbol, DateRange, Horizon, MetricName`

- `from vip.domain.protocols import MarketDataSource, VolatilityModel`

## Notes

- Keep this layer independent and highly testable.

- If a new feature requires vendor/library-specific logic, it belongs in another package and should conform to domain protocols.