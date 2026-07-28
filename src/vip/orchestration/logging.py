"""Logging configuration helpers for VIP processes.

Exports
-------
configure_logging
    Configure the root logger once for a process.
get_logger
    Return a named logger.
"""

from __future__ import annotations

import logging


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging if it has not been configured yet.

    Parameters
    ----------
    level : str, default ``'INFO'``
        Stdlib logging level name.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__``.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logging.getLogger(name)