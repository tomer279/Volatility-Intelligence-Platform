# `vip.config`

## Purpose

The config package loads and validates application settings from YAML into typed objects.  

It is the single source of truth for runtime settings used by CLI commands and pipelines.

## Modules

- `schema.py` - Pydantic models for validated application config.

- `defaults.py` - Helpers for resolving project root and default config file path.

- `loader.py` - YAML read + validation entrypoint `load_config`).

- `__init__.py` - Public config exports.

## Key APIs

- `AppConfig` - Top-level validated configuration model.

- `load_config(path=None)` - Loads YAML and returns `AppConfig`.

- `default_config_path()` - Resolves `configs/default.yaml`.

- `resolve_project_root()` - Locates repo root via `pyproject.toml`.

## Dependencies

- Depends on: `pydantic`, `pyyaml`, domain enums/errors.

- Must not depend on: ingestion, modeling, or persistence implementations.

## Usage

Example pattern:

1. CLI/pipeline calls `load_config()`.

2. Receives `AppConfig`.

3. Passes typed fields to downstream use-cases.

## Notes

- YAML files should remain human-readable and diff-friendly.

- Validation failures should raise `ConfigError` with clear, actionable messages.