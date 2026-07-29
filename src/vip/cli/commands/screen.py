"""CLI command for factor screening.

Exports
-------
screen_command
    Register the ``vip screen`` command on a Typer app.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vip.application.screen_factors import ScreenConfig, screen_factors
from vip.config import load_config, resolve_project_root
from vip.domain.value_objects import Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO = 5
DEFAULT_N_REPEATS = 5
DEFAULT_TOP_K = 3


def screen_command(app: typer.Typer) -> None:
    """Register the factor-screening command.

    Parameters
    ----------
    app : typer.Typer
        Target CLI app instance.
    """

    @app.command("screen")
    def screen(
        symbol: str | None = typer.Option(
            None,
            "--symbol",
            help="Ticker symbol override (defaults to config symbol).",
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
        n_repeats: int = typer.Option(
            DEFAULT_N_REPEATS,
            "--n-repeats",
            help="Permutation repeats per feature per fold.",
        ),
        top_k: int = typer.Option(
            DEFAULT_TOP_K,
            "--top-k",
            help="Top-k used for factor hit-rate stability.",
        ),
    ) -> None:
        """Run factor screening and print a ranked factor table."""
        config = load_config()
        effective_symbol = (
            Symbol(symbol) if symbol is not None else Symbol(config.symbol)
        )
        project_root = resolve_project_root()

        processed_dir = Path(config.paths.processed_dir)
        if not processed_dir.is_absolute():
            processed_dir = project_root / processed_dir

        artifacts_dir = Path(config.paths.artifacts_dir)
        if not artifacts_dir.is_absolute():
            artifacts_dir = project_root / artifacts_dir

        feature_store = ParquetFeatureMatrixStore(root_dir=processed_dir)
        artifact_store = FilesystemArtifactStore(root_dir=artifacts_dir)

        result = screen_factors(
            feature_store=feature_store,
            artifact_store=artifact_store,
            symbol=effective_symbol,
            config=ScreenConfig(
                n_splits=n_splits,
                embargo_size=embargo,
                n_repeats=n_repeats,
                top_k=top_k,
            ),
        )

        experiment_dir = artifacts_dir / result.experiment_id.as_path_key()
        typer.echo(f"Factor screen results ({result.symbol.value})")
        typer.echo(f"screening_model: {result.screening_model}")
        typer.echo("metric primary: qlike (lower is better)")
        typer.echo("")
        typer.echo("Model horse-race")
        typer.echo(result.summary.to_string(index=False))
        typer.echo("")
        typer.echo("Ranked factors (mean ΔQLIKE importance)")
        typer.echo(result.ranking.to_string(index=False))
        typer.echo("")
        typer.echo(f"top_feature: {result.top_feature()}")
        typer.echo(f"best_model_qlike: {result.best_model_qlike():.6f}")
        typer.echo(f"report: {experiment_dir / 'report.html'}")
