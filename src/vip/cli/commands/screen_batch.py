"""CLI command for multi-symbol batch screening."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from vip.application.screen_batch import BatchScreenConfig, run_screen_batch
from vip.application.screen_factors import ScreenConfig
from vip.config import load_config, resolve_project_root
from vip.domain.value_objects import DateRange, Symbol
from vip.ingestion.yfinance_source import YFinanceMarketDataSource
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore

DEFAULT_SYMBOLS = "SPY,QQQ,IWM"

DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO = 5
DEFAULT_N_REPEATS = 5
DEFAULT_TOP_K = 3



def screen_batch_command(app: typer.Typer) -> None:
    """Register the ``vip screen-batch`` command.

    Parameters
    ----------
    app : typer.Typer
        Target CLI app instance.
    """

    @app.command("screen-batch")
    def screen_batch(
        symbols: str = typer.Option(
            DEFAULT_SYMBOLS,
            "--symbols",
            help="Comma-separated symbols to screen (default: SPY,QQQ,IWM).",
        ),
        skip_ingest: bool = typer.Option(
            False,
            "--skip-ingest",
            help="Skip ingest and fail if market data is missing for any symbol.",
        ),
        skip_features: bool = typer.Option(
            False,
            "--skip-features",
            help="Skip feature-building and fail if feature matrices are missing.",
        ),
        n_splits: int = typer.Option(
            DEFAULT_N_SPLITS,
            "--n-splits",
            help="Number of expanding walk-forward folds.",
        ),
        embargo: int = typer.Option(
            DEFAULT_EMBARGO,
            "--embargo",
            help="Embargo size in trading sessions between train and test.",
        ),
    ) -> None:
        """Run multi-symbol batch screening and print a summary table."""
        config = load_config()
        project_root = resolve_project_root()

        artifact_store = FilesystemArtifactStore(
            root_dir=_resolve_dir(config.paths.artifacts_dir, project_root),
        )

        batch_cfg = BatchScreenConfig(
            symbols=_parse_symbols(symbols),
            skip_ingest=skip_ingest,
            skip_features=skip_features,
            date_range=DateRange(
                start=config.date_range.start,
                end=config.date_range.end or date.today(),
            ),
            horizon_days=config.target.horizon_days,
            screen_config=ScreenConfig(
                n_splits=n_splits,
                embargo_size=embargo,
            ),
        )

        result = run_screen_batch(
            source=YFinanceMarketDataSource(),
            market_store=ParquetMarketDataStore(
                root_dir=_resolve_dir(config.paths.raw_dir, project_root),
            ),
            feature_store=ParquetFeatureMatrixStore(
                root_dir=_resolve_dir(config.paths.processed_dir, project_root),
            ),
            artifact_store=artifact_store,
            config=batch_cfg,
        )

        typer.echo("Batch screen results")
        typer.echo(result.summary.to_string(index=False))
        typer.echo("")
        for sym, exp_id in result.experiments.items():
            typer.echo(
                f"{sym.value}: {artifact_store.experiment_dir(exp_id) / 'report.html'}"
            )


def _resolve_dir(raw: str, project_root: Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(raw)
    if not path.is_absolute():
        return project_root / path
    return path


def _parse_symbols(raw: str) -> list[Symbol]:
    """Split a comma-separated string into Symbol objects."""
    return [Symbol(s.strip()) for s in raw.split(",") if s.strip()]
