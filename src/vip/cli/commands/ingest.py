"""CLI command for market data ingestion.

Exports
-------
ingest_command
    Register the ``vip ingest`` command on a Typer app.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from vip.application.ingest_market_data import ingest_market_data
from vip.config import load_config, resolve_project_root
from vip.domain.value_objects import DateRange, Symbol
from vip.ingestion.yfinance_source import YFinanceMarketDataSource
from vip.persistence.parquet_store import ParquetMarketDataStore


def _parse_iso_date(value: str, option_name: str) -> date:
    """Parse an ISO date string from a CLI option.

    Parameters
    ----------
    value : str
        Date text in ``YYYY-MM-DD`` form.
    option_name : str
        CLI option name used in error messages.

    Returns
    -------
    datetime.date
        Parsed calendar date.

    Raises
    ------
    typer.BadParameter
        If ``value`` is not a valid ISO date.
    """
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{option_name} must be YYYY-MM-DD, got: {value!r}"
        ) from exc


def ingest_command(app: typer.Typer) -> None:
    """Register the market data ingestion command.

    Parameters
    ----------
    app : typer.Typer
        Target CLI app instance.
    """
    @app.command("ingest")
    def ingest(
        symbol: str | None = typer.Option(
            None,
            "--symbol",
            help="Ticker symbol override (defaults to config symbol).",
        ),
        start: str | None = typer.Option(
            None,
            "--start",
            help="Start date override in YYYY-MM-DD format.",
        ),
        end: str | None = typer.Option(
            None,
            "--end",
            help="End date override in YYYY-MM-DD format.",
        ),
    ) -> None:
        """Fetch, validate, and persist daily OHLCV market data."""
        config = load_config()

        effective_symbol = (
            Symbol(symbol) if symbol is not None else Symbol(config.symbol)
        )

        start_date = (
            _parse_iso_date(start, "--start")
            if start is not None
            else config.date_range.start
        )
        if end is not None:
            end_date = _parse_iso_date(end, "--end")
        elif config.date_range.end is not None:
            end_date = config.date_range.end
        else:
            end_date = date.today()

        date_range = DateRange(start=start_date, end=end_date)

        raw_dir = Path(config.paths.raw_dir)
        if not raw_dir.is_absolute():
            raw_dir = resolve_project_root() / raw_dir

        source = YFinanceMarketDataSource()
        store = ParquetMarketDataStore(root_dir=raw_dir)

        result = ingest_market_data(
            source=source,
            store=store,
            symbol=effective_symbol,
            date_range=date_range,
        )

        typer.echo("Ingestion completed.")
        typer.echo(f"source: {source.source_name()}")
        typer.echo(f"symbol: {result.symbol.value}")
        typer.echo(f"rows: {result.row_count}")
        typer.echo(f"dates: {result.date_span_label()}")
        typer.echo(f"output: {result.output_path}")
