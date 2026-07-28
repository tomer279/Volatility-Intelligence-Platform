"""Default configuration path helpers.

Exports
-------
resolve_project_root
    Locate the repository root directory.
default_config_path
    Resolve the path to ``configs/default.yaml``.
"""

from __future__ import annotations

from pathlib import Path

from vip.domain.errors import ConfigError


def resolve_project_root() -> Path:
    """Locate the repository root containing ``pyproject.toml``.

    Returns
    -------
    pathlib.Path
        Absolute project root path.

    Raises
    ------
    ConfigError
        If no ``pyproject.toml`` is found in parent directories.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError("Could not locate project root (pyproject.toml).")


def default_config_path() -> Path:
    """Return the path to the checked-in default YAML config.

    Returns
    -------
    pathlib.Path
        Absolute path to ``configs/default.yaml``.
    """
    return resolve_project_root() / "configs" / "default.yaml"