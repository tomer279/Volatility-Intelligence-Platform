"""CLI command for the composite study pipeline.

Exports
-------
run_command
    Register the ``vip run`` command on a Typer app.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from vip.application.run_study import (
    RunStudyConfig,
    RunStudyStores,
    run_study,
    FeatureMatrixExtras
)
from vip.application.screen_batch import BatchScreenResult
from vip.config import load_config, resolve_project_root
from vip.cli.feature_extras import parse_feature_extras
from vip.domain.value_objects import DateRange, Symbol
from vip.ingestion.yfinance_source import YFinanceMarketDataSource
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore


def run_command(app: typer.Typer) -> None:
    """Register the ``vip run`` command.

    Parameters
    ----------
    app : typer.Typer
        Target CLI app instance.
    """

    @app.command("run")
    def run(
            symbols: str | None = typer.Option(
                None,
                "--symbols",
                help="Comma-separated symbols (e.g. SPY,QQQ).",
            ),
            symbol: str | None = typer.Option(
                None,
                "--symbol",
                help="Single-symbol shorthand (mutually exclusive with --symbols).",
            ),
            with_features: str = typer.Option(
                "",
                "--with",
                help="Comma-separated extras: vix, jump (e.g. vix,jump).",
            ),
            skip_ingest: bool = typer.Option(
                False,
                "--skip-ingest",
                help="Skip ingestion; fail if data is missing.",
            ),
            skip_features: bool = typer.Option(
                False,
                "--skip-features",
                help="Skip feature building; fail if missing.",
            ),
    ) -> None:
        """Run the full ingest → features → screen pipeline."""
        parsed_symbols = _resolve_symbols(symbols, symbol)
        config = load_config()
        project_root = resolve_project_root()
        stores = _build_stores(config, project_root)
        extras = parse_feature_extras(with_features)
        study_cfg = _build_study_config(
            parsed_symbols,
            config,
            extras,
            skip_ingest,
            skip_features,
        )
        result = run_study(stores, study_cfg)
        _print_results(result, stores.artifact_store)


def _build_study_config(
        symbols: list[Symbol],
        config,
        extras: FeatureMatrixExtras,
        skip_ingest: bool,
        skip_features: bool,
) -> RunStudyConfig:
    """Build a ``RunStudyConfig`` from CLI flags and app config."""
    return RunStudyConfig(
        symbols=symbols,
        date_range=DateRange(
            start=config.date_range.start,
            end=config.date_range.end or date.today(),
        ),
        horizon_days=config.target.horizon_days,
        extras=extras,
        skip_ingest=skip_ingest,
        skip_features=skip_features,
    )


def _print_results(
        result: BatchScreenResult,
        artifact_store: FilesystemArtifactStore,
) -> None:
    """Print study summary table and report paths."""
    typer.echo("Study results")
    typer.echo(result.summary.to_string(index=False))
    typer.echo("")
    for sym, exp_id in result.experiments.items():
        report = artifact_store.experiment_dir(exp_id) / "report.html"
        typer.echo(f"{sym.value}: {report}")


def _resolve_symbols(
        symbols: str | None,
        symbol: str | None,
) -> list[Symbol]:
    """Parse ``--symbols`` / ``--symbol`` with mutual exclusivity."""
    if symbols is not None and symbol is not None:
        raise typer.BadParameter(
            "Use --symbols or --symbol, not both."
        )
    if symbol is not None:
        return [Symbol(symbol.strip())]
    if symbols is not None:
        return [Symbol(s.strip()) for s in symbols.split(",") if s.strip()]
    raise typer.BadParameter(
        "Provide --symbol or --symbols."
    )


def _build_stores(config, project_root: Path) -> RunStudyStores:
    """Construct persistence stores from configuration."""
    return RunStudyStores(
        source=YFinanceMarketDataSource(),
        market_store=ParquetMarketDataStore(
            root_dir=_resolve_dir(config.paths.raw_dir, project_root),
        ),
        feature_store=ParquetFeatureMatrixStore(
            root_dir=_resolve_dir(config.paths.processed_dir, project_root),
        ),
        artifact_store=FilesystemArtifactStore(
            root_dir=_resolve_dir(config.paths.artifacts_dir, project_root),
        ),
    )


def _resolve_dir(raw: str, project_root: Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(raw)
    if not path.is_absolute():
        return project_root / path
    return path
