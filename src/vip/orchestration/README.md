# `vip.orchestration`

## Purpose

The orchestration package contains process-level coordination helpers used across commands and pipelines.  

In Milestone 0 it only includes logging setup; dependency wiring/container patterns are added later.

## Modules

- `logging.py` - Shared logging configuration and logger factory.

- `__init__.py` - Public orchestration exports.

## Key APIs

- `configure_logging(level="INFO")` - One-time process logging setup.

- `get_logger(name)` - Retrieve module-specific logger instance.

## Dependencies

- Depends on: stdlib `logging`.

- Must not depend on: business logic from ingestion/features/modeling/evaluation.

## Usage

Typical pattern:

1. CLI callback calls `configure_logging(...)`.

2. Modules call `get_logger(__name__)`.

3. Logs become consistent across the application.

## Notes

- Keep orchestration thin.

- Avoid placing business logic here; this package should focus on runtime wiring concerns.