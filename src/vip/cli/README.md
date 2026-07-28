# `vip.cli`

## Purpose

The CLI package exposes user-facing commands for operating the research platform from the terminal.  

It should remain a thin interface layer that delegates work to application/use-case modules.

## Modules

- `main.py` - Typer app entrypoint, global options, and initial commands.

- `__init__.py` - Exposes the CLI app object.

- `commands/` - (planned) command modules such as ingest, features, train, evaluate, report.

## Key APIs

- `app` - Typer application referenced by `pyproject.toml` script entrypoint.

- `info` command - Prints version and selected default config values.

- Root callback - Applies shared runtime options (for example log level).

## Dependencies

- Depends on: `typer`, config loader, orchestration logging.

- Must not depend on: concrete vendor APIs or heavy modeling logic directly.

## Usage

From an installed editable environment:

- `vip --help`

- `vip info`

Future:

- `vip ingest --symbol SPY`

- `vip train --config configs/experiments/...`

## Notes

- Keep commands composable and explicit.

- Prefer passing configuration paths/flags rather than hardcoding behavior.

- Delegate business logic to application modules to preserve testability.