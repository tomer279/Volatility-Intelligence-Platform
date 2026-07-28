"""Load and validate VIP YAML configuration files.

Exports
-------
load_config
    Load YAML from disk into an ``AppConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from vip.config.defaults import default_config_path
from vip.config.schema import AppConfig
from vip.domain.errors import ConfigError


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate an application config file.

    Parameters
    ----------
    path : pathlib.Path or None, default None
        YAML file to load. When ``None``, uses ``configs/default.yaml``.

    Returns
    -------
    AppConfig
        Validated configuration object.

    Raises
    ------
    ConfigError
        If the file is missing, unreadable, or fails schema validation.
    """
    config_path = path if path is not None else default_config_path()
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = _read_yaml(config_path)
    except OSError as exc:
        raise ConfigError(f"Could not read config file: {config_path}") from exc

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config at {config_path}: {exc}") from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML mapping from disk.

    Parameters
    ----------
    path : pathlib.Path
        YAML file path.

    Returns
    -------
    dict of str to Any
        Parsed mapping.

    Raises
    ------
    ConfigError
        If the document is not a mapping.
    """
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return loaded