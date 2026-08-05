# `vip.cli`

## Purpose
The CLI package exposes user-facing commands for operating the research platform from the terminal.  
It should remain a thin interface layer that delegates work to application/use-case modules.

## Modules
- `main.py` - Typer app entrypoint, global options, and command registration.
- `__init__.py` - Exposes the CLI app object.
- `commands/ingest.py` - Registers the `vip ingest` command.
- `commands/features.py` - Registers the `vip features` command.
- `commands/evaluate.py` - Registers the `vip evaluate` command.
- `commands/screen.py` - Registers the `vip screen` command.
- `commands/__init__.py` - Re-exports command registration helpers.
- `commands/screen_batch.py` - Registers the `vip screen-batch` command.
- `commands/screen_multi_horizon.py` - Registers the `vip screen-horizons` command.
- `commands/run.py` - Registers the `vip run` command.
- `feature_extras.py` - Shared `--with` parser → `FeatureMatrixExtras`.

## Key APIs
- `app` - Typer application referenced by the `pyproject.toml` script entrypoint.
- Root callback - Applies shared runtime options (for example log level).
- `info` command - Prints package version and selected default config values.
- `ingest` command - Fetches, validates, and persists daily OHLCV data.
- `features` command - Builds and persists a feature matrix (`--symbol`, `--horizon`, `--with`).
- `evaluate` command - Runs baseline walk-forward evaluation and prints a comparison table.
- `screen` command - Runs factor screening (horse-race + inference vs HAR + Ridge importance)
  and writes an HTML research memo.
- `ingest_command(app)` / `features_command(app)` / `evaluate_command(app)` / `screen_command(app)` - Command registrars.
- `screen-batch` command - Runs multi-symbol screening with `--skip-ingest` / `--skip-features` flags.
- `screen_batch_command(app)` - Command registrar.
- `screen-horizons` command - Multi-horizon study (`--symbol`, `--horizons`, `--with`, `--skip-features`).
- `screen_multi_horizon_command(app)` - Command registrar.
- `run` command - One-shot ingest → features → screen (`--symbol` or `--symbols`, `--with`, skip flags).
- `parse_feature_extras(raw)` - Parse `--with` tokens (`vix`, `jump`) into `FeatureMatrixExtras`.
- `run_command(app)` - Command registrar.

## Dependencies
- Depends on: `typer`, config loader, orchestration logging, application use-cases, persistence stores.
- Prefer keeping vendor/SDK details out of CLI modules when practical; construct adapters in the command and call application use-cases.

## Usage
From an installed editable environment:

- `vip --help`
- `vip info`
- `vip ingest --symbol SPY --start 2018-01-01 --end 2024-12-31`
- `vip features --symbol SPY`
- `vip features --symbol SPY --with vix,jump`
- `vip evaluate --help`
- `vip evaluate --symbol SPY`
- `vip evaluate --symbol SPY --n-splits 5 --embargo 5`
- `vip screen --help`
- `vip screen --symbol SPY`
- `vip screen --symbol SPY --n-splits 5 --embargo 5 --n-repeats 5 --top-k 3`
- `vip screen-horizons --help`
- `vip screen-horizons --symbol SPY --with vix`
- `vip screen-horizons --symbol SPY --with vix,jump`
- `vip screen-horizons --symbol SPY --horizons 1,5,21 --skip-features`
- `vip screen-batch --symbols SPY,QQQ,IWM`
- `vip screen-batch --symbols SPY,QQQ --skip-ingest --skip-features`
- `vip run --symbol SPY --with vix`
- `vip run --symbols SPY,QQQ --with vix`
- `vip run --symbol SPY --with vix,jump`
- `vip run --symbol SPY --skip-ingest --skip-features`

## Notes
- Keep commands composable and explicit.
- Prefer CLI flags/config overrides over hardcoding behavior.
- Date options are strings (`YYYY-MM-DD`) because Typer does not support `datetime.date` annotations.
- `features` reads OHLCV from `data/raw/` and writes matrices to `data/processed/`.
- `evaluate` reads `data/processed/` and writes artifacts under `data/artifacts/`.
- `screen` reads `data/processed/`, prints horse-race + inference (mean ΔQLIKE, bootstrap CI/p)
  + ranked factors, and writes artifacts under `data/artifacts/` including `oos_losses.json`,
  `inference.json`, optional `inference_sensitivity.json`, and `report.html`.
- Delegate business logic to application modules to preserve testability.
- `--with` accepts comma-separated tokens `vix` and/or `jump` (see `feature_extras.parse_feature_extras`).
- `run` orchestrates ingest (including VIX when `--with` contains `vix`), features, and screening; prints report paths under `data/artifacts/`.
- `screen-horizons` rebuilds features per horizon unless `--skip-features`; writes
  `data/artifacts/multi-horizon-screen-{symbol}-{date}/` including `horizon_summary.json`
  and study-level `report.html`. `vip screen` remains the single-horizon entrypoint (default h=5).
  