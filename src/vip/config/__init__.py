"""Configuration schema and loading utilities.

Exports
-------
AppConfig
    Top-level validated settings model.
load_config
    Load YAML into ``AppConfig``.
default_config_path
    Path to the default checked-in YAML.
resolve_project_root
    Repository root discovery helper.
"""

from vip.config.defaults import default_config_path, resolve_project_root
from vip.config.loader import load_config
from vip.config.schema import AppConfig

__all__ = [
    "AppConfig",
    "default_config_path",
    "load_config",
    "resolve_project_root",
]