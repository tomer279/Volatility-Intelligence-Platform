"""Command-line interface for the Volatility Intelligence Platform.

Exports
-------
app
    Typer application entrypoint.
"""

from __future__ import annotations

import typer

from vip import __version__
from vip.config import load_config
from vip.orchestration import configure_logging
from vip.cli.commands import (
    evaluate_command,
    features_command,
    ingest_command,
    screen_command,
    screen_batch_command
)

app = typer.Typer(
    name="vip",
    help="Volatility Intelligence Platform",
    no_args_is_help=True,
)
ingest_command(app)
features_command(app)
evaluate_command(app)
screen_command(app)
screen_batch_command(app)

@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    ),
) -> None:
    """Configure process-wide options for VIP commands."""
    configure_logging(log_level)


@app.command("info")
def info() -> None:
    """Print package version and default configuration summary."""
    config = load_config()
    typer.echo(f"vip {__version__}")
    typer.echo(f"symbol: {config.symbol}")
    typer.echo(f"horizon_days: {config.target.horizon_days}")
    typer.echo(f"rv_estimator: {config.target.rv_estimator.value}")
    typer.echo(f"primary_metric: {config.evaluation.primary_metric.value}")


if __name__ == "__main__":
    app()
