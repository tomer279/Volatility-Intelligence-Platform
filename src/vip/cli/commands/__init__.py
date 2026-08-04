"""CLI command registration helpers.

Exports
-------
ingest_command
    Register the ``vip ingest`` command.
features_command
    Register the ``vip features`` command.
evaluate_command
    Register the ``vip evaluate`` command.
screen_command
    Register the ``vip screen`` command.
screen_batch_command
    Register the ``vip screen-batch`` command.
screen_multi_horizon_command
    Register the ``vip screen-horizons`` command.
run_command
    Register the ``vip run`` command.
"""

from vip.cli.commands.evaluate import evaluate_command
from vip.cli.commands.features import features_command
from vip.cli.commands.ingest import ingest_command
from vip.cli.commands.run import run_command
from vip.cli.commands.screen import screen_command
from vip.cli.commands.screen_batch import screen_batch_command
from vip.cli.commands.screen_multi_horizon import screen_multi_horizon_command


__all__ = [
    "evaluate_command",
    "features_command",
    "ingest_command",
    "run_command",
    "screen_command",
    "screen_batch_command",
    "screen_multi_horizon_command",
]