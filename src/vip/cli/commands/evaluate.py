"""CLI command for baseline walk-forward evaluation.

Exports
-------
evaluate_command
    Register the ``vip evaluate`` command on a Typer app.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vip.application.run_baseline_experiment import run_baseline_experiment
from vip.config import load_config, resolve_project_root
from vip.domain.value_objects import Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO = 5


def evaluate_command(app: typer.Typer) -> None:
    """Register the baseline evaluation command.

    Parameters
    ----------
    app : typer.Typer
        Target CLI app instance.
    """

    @app.command("evaluate")
    def evaluate(
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
    ) -> None:
        """Run baseline walk-forward evaluation and print a comparison table."""
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

        result = run_baseline_experiment(
            feature_store=feature_store,
            artifact_store=artifact_store,
            symbol=effective_symbol,
            n_splits=n_splits,
            embargo_size=embargo,
        )

        typer.echo(
            f"Baseline walk-forward results ({result.symbol.value})"
        )
        typer.echo("metric primary: qlike (lower is better)")
        typer.echo("")
        typer.echo(result.summary.to_string(index=False))
        typer.echo("")
        typer.echo(f"best_model: {result.primary_model}")
        typer.echo(f"best_qlike: {result.best_qlike():.6f}")
        typer.echo(
            "artifact: "
            f"{artifacts_dir / result.experiment_id.as_path_key() / 'metrics.json'}"
        )
