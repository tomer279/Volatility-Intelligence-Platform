"""Command-line interface package for VIP.

Exports
-------
app
    Typer application from ``vip.cli.main``.
"""

from vip.cli.main import app

__all__ = ["app"]