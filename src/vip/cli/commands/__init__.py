"""CLI command registration helpers.

Exports
-------
ingest_command
    Register the ``vip ingest`` command.
"""

from vip.cli.commands.ingest import ingest_command

__all__ = ["ingest_command"]