"""CLI command for feature-matrix construction.

Exports
-------
features_command
    Register the ``vip features`` command on a Typer app.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vip.application.build_feature_matrix import (
    FeatureMatrixExtras,
    build_and_persist_feature_matrix,
)
from vip.config import load_config, resolve_project_root
from vip.domain.value_objects import Symbol
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore


def features_command(app: typer.Typer) -> None:
    """Register the feature-matrix build command.

    Parameters
    ----------
    app : typer.Typer
        Target CLI app instance.
    """

    @app.command("features")
    def features(
        symbol: str | None = typer.Option(
            None, "--symbol", help="Ticker symbol override."
        ),
        horizon: int | None = typer.Option(
            None, "--horizon", help="Horizon override in days."
        ),
        with_vix: bool = typer.Option(
            False,
            "--with-vix",
            help="Join VIX level/change features (requires vip ingest --symbol VIX).",
        ),
    ) -> None:
        """Build and persist a feature matrix from ingested OHLCV data."""
        config = load_config()

        effective_symbol = (
            Symbol(symbol) if symbol is not None else Symbol(config.symbol)
        )
        horizon_days = (
            horizon if horizon is not None else config.target.horizon_days
        )

        project_root = resolve_project_root()

        raw_dir = Path(config.paths.raw_dir)
        if not raw_dir.is_absolute():
            raw_dir = project_root / raw_dir

        processed_dir = Path(config.paths.processed_dir)
        if not processed_dir.is_absolute():
            processed_dir = project_root / processed_dir

        market_store = ParquetMarketDataStore(root_dir=raw_dir)
        feature_store = ParquetFeatureMatrixStore(root_dir=processed_dir)

        result = build_and_persist_feature_matrix(
            market_store=market_store,
            feature_store=feature_store,
            symbol=effective_symbol,
            horizon_days=horizon_days,
            extras=FeatureMatrixExtras(include_vix=with_vix),
        )

        typer.echo("Feature matrix build completed.")
        typer.echo(f"symbol: {result.symbol.value}")
        typer.echo(f"horizon_days: {horizon_days}")
        typer.echo(f"rows: {result.row_count}")
        typer.echo(f"features: {result.feature_count}")
        typer.echo(f"dates: {result.date_span_label()}")
        typer.echo(f"output: {result.output_path}")
