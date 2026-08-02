"""CLI command for factor screening.

Exports
-------
screen_command
    Register the ``vip screen`` command on a Typer app.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vip.application.screen_factors import (
    FactorScreenResult,
    ScreenConfig,
    screen_factors,
)
from vip.config import load_config, resolve_project_root
from vip.domain.value_objects import Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO = 5
DEFAULT_N_REPEATS = 5
DEFAULT_TOP_K = 3
BASELINE_MODEL = "har_rv_ols"
INFERENCE_COLUMNS: tuple[str, ...] = (
    "model",
    "mean_delta_qlike",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_pvalue",
    "significant_vs_baseline",
)


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
        """Run factor screening and print horse-race, inference, and factor tables."""
        config = load_config()
        effective_symbol = (
            Symbol(symbol) if symbol is not None else Symbol(config.symbol)
        )
        feature_store, artifact_store = _build_screen_stores(config)
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
        _print_screen_results(result, artifact_store)


def _build_screen_stores(config) -> tuple[
    ParquetFeatureMatrixStore, FilesystemArtifactStore
]:
    """Build feature and artifact stores from loaded config.

    Parameters
    ----------
    config
        Loaded VIP configuration.

    Returns
    -------
    tuple of ParquetFeatureMatrixStore and FilesystemArtifactStore
        Persistence stores for the screen run.
    """
    project_root = resolve_project_root()
    feature_store = ParquetFeatureMatrixStore(
        root_dir=_resolve_dir(config.paths.processed_dir, project_root),
    )
    artifact_store = FilesystemArtifactStore(
        root_dir=_resolve_dir(config.paths.artifacts_dir, project_root),
    )
    return feature_store, artifact_store


def _resolve_dir(raw: str, project_root: Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(raw)
    if not path.is_absolute():
        return project_root / path
    return path


def _print_screen_results(
    result: FactorScreenResult,
    artifact_store: FilesystemArtifactStore,
) -> None:
    """Print horse-race, ranking, inference, and report path.

    Parameters
    ----------
    result : FactorScreenResult
        Completed screen outputs.
    artifact_store : FilesystemArtifactStore
        Store used to resolve the HTML report path.
    """
    report_path = (
        artifact_store.experiment_dir(result.identity.experiment_id)
        / "report.html"
    )
    typer.echo(f"Factor screen results ({result.identity.symbol.value})")
    typer.echo(f"screening_model: {result.identity.screening_model}")
    typer.echo("metric primary: qlike (lower is better)")
    typer.echo("")
    typer.echo("Model horse-race")
    typer.echo(result.tables.summary.to_string(index=False))
    typer.echo("")
    typer.echo("Ranked factors (median ΔQLIKE importance)")
    typer.echo(result.tables.ranking.to_string(index=False))
    typer.echo("")
    _print_inference_table(result.tables.summary)
    typer.echo(f"top_feature: {result.top_feature()}")
    typer.echo(f"best_model_qlike: {result.best_model_qlike():.6f}")
    typer.echo(f"report: {report_path}")


def _print_inference_table(summary) -> None:
    """Print challenger bootstrap inference rows when columns exist.

    Parameters
    ----------
    summary : pandas.DataFrame
        Horse-race summary, optionally inference-enriched.
    """
    typer.echo(f"Inference vs {BASELINE_MODEL} (block bootstrap primary)")
    if not all(col in summary.columns for col in INFERENCE_COLUMNS):
        typer.echo("(inference columns unavailable)")
        typer.echo("")
        return
    challengers = summary.loc[
        summary["model"].astype(str) != BASELINE_MODEL,
        list(INFERENCE_COLUMNS),
    ]
    typer.echo(challengers.to_string(index=False))
    typer.echo("")
