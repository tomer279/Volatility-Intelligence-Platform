# `vip.cli`

## Purpose
The CLI package exposes user-facing commands for operating the research platform from the terminal.  
It should remain a thin interface layer that delegates work to application/use-case modules.

## Modules
- `main.py` - Typer app entrypoint, global options, and command registration.
- `__init__.py` - Exposes the CLI app object.
- `commands/ingest.py` - Registers the `vip ingest` command.
- `commands/features.py` - Registers the `vip features` command.
- `commands/__init__.py` - Re-exports command registration helpers.

## Key APIs
- `app` - Typer application referenced by the `pyproject.toml` script entrypoint.
- Root callback - Applies shared runtime options (for example log level).
- `info` command - Prints package version and selected default config values.
- `ingest` command - Fetches, validates, and persists daily OHLCV data.
- `features` command - Builds and persists a feature matrix from ingested OHLCV.
- `ingest_command(app)` - Registers the ingest command on the Typer app.
- `features_command(app)` - Registers the features command on the Typer app.

## Dependencies
- Depends on: `typer`, config loader, orchestration logging, application use-cases, persistence stores.
- Prefer keeping vendor/SDK details out of CLI modules when practical; construct adapters in the command and call application use-cases.

## Usage
From an installed editable environment:

- `vip --help`
- `vip info`
- `vip ingest --help`
- `vip ingest --symbol SPY --start 2018-01-01 --end 2024-12-31`
- `vip features --help`
- `vip features --symbol SPY`
- `vip features --symbol SPY --horizon 5`

Future:

- `vip train --config configs/experiments/...`
- `vip evaluate ...`
- `vip report ...`

## Notes
- Keep commands composable and explicit.
- Prefer CLI flags/config overrides over hardcoding behavior.
- Date options are strings (`YYYY-MM-DD`) because Typer does not support `datetime.date` annotations.
- `features` reads OHLCV from `data/raw/` and writes matrices to `data/processed/`.
- Delegate business logic to application modules to preserve testability.