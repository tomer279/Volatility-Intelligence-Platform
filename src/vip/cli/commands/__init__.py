"""CLI command registration helpers.

Exports
-------
ingest_command
    Register the ``vip ingest`` command.
features_command
    Register the ``vip features`` command.
"""

from vip.cli.commands.features import features_command
from vip.cli.commands.ingest import ingest_command

__all__ = [
    "features_command",
    "ingest_command",
]