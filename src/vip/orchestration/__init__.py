"""Orchestration helpers (logging and, later, dependency wiring).

Exports
-------
configure_logging
    Configure process logging.
get_logger
    Return a named logger.
"""

from vip.orchestration.logging import configure_logging, get_logger

__all__ = [
    "configure_logging",
    "get_logger",
]