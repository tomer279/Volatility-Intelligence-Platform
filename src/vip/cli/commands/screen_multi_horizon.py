"""CLI command for multi-horizon factor screening.

Exports
-------
screen_multi_horizon_command
    Register the ``vip screen-horizons`` command on a Typer app.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vip.application.screen_factors import ScreenConfig
from vip.application.screen_multi_horizon import (
    MultiHorizonScreenConfig,
    MultiHorizonScreenResult,
    MultiHorizonStores,
    screen_multi_horizon,
)
from vip.config import load_config, resolve_project_root
from vip.domain.value_objects import Symbol
from vip.evaluation.horizon_defaults import LOCKED_SCREEN_HORIZONS
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore


_PRINT_COLUMNS: tuple[str, ...] = (
    "horizon_days",
    "model",
    "qlike",
    "mean_delta_qlike",
    "bootstrap_pvalue",
    "significant_vs_baseline",
)


def screen_multi_horizon_command(app: typer.Typer) -> None:
    """Register the multi-horizon factor-screening command.

    Parameters
    ----------
    app : typer.Typer
        Target CLI app instance.
    """

    @app.command("screen-horizons")
    def screen_horizons(
        symbol: str | None = typer.Option(
            None,
            "--symbol",
            help="Ticker symbol override (defaults to config symbol).",
        ),
        horizons: str = typer.Option(
            "1,5,21",
            "--horizons",
            help="Comma-separated horizons (locked set: 1,5,21).",
        ),
        with_vix: bool = typer.Option(
            False,
            "--with-vix",
            help="Join VIX features when (re)building per-horizon matrices.",
        ),
        skip_features: bool = typer.Option(
            False,
            "--skip-features",
            help="Skip feature builds; require target_rv_cc_{h}d per horizon.",
        ),
    ) -> None:
        """Run multi-horizon screens and print the cross-horizon summary."""
        config = load_config()
        project_root = resolve_project_root()
        effective_symbol = (
            Symbol(symbol) if symbol is not None else Symbol(config.symbol)
        )
        stores = _build_stores(config, project_root)
        result = screen_multi_horizon(
            stores=stores,
            config=MultiHorizonScreenConfig(
                symbol=effective_symbol,
                horizons=_parse_horizons(horizons),
                with_vix=with_vix,
                skip_features=skip_features,
                screen_config=ScreenConfig(),
            ),
        )
        _print_screen_horizons_result(result, stores, effective_symbol)


def _build_stores(config, project_root: Path) -> MultiHorizonStores:
    """Build market / feature / artifact stores from loaded config."""
    return MultiHorizonStores(
        market_store=ParquetMarketDataStore(
            root_dir=_resolve_dir(config.paths.raw_dir, project_root),
        ),
        feature_store=ParquetFeatureMatrixStore(
            root_dir=_resolve_dir(config.paths.processed_dir, project_root),
        ),
        artifact_root=_resolve_dir(config.paths.artifacts_dir, project_root),
    )


def _print_screen_horizons_result(
        result: MultiHorizonScreenResult,
        stores: MultiHorizonStores,
        symbol: Symbol,
) -> None:
    """Print the cross-horizon summary and artifact paths."""
    study_dir = stores.resolve_study_dir(result.study_id)
    typer.echo(f"Multi-horizon screen ({symbol.value})")
    typer.echo("metric primary: qlike (lower is better)")
    typer.echo("")
    printable = result.summary.loc[
        :,
        [c for c in _PRINT_COLUMNS if c in result.summary.columns],
    ]
    typer.echo(printable.to_string(index=False))
    typer.echo("")
    typer.echo(f"study: {study_dir}")
    typer.echo(f"summary: {study_dir / 'horizon_summary.json'}")
    typer.echo(f"report: {study_dir / 'report.html'}")


def _parse_horizons(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated horizons string into a tuple of ints."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return LOCKED_SCREEN_HORIZONS
    return tuple(int(part) for part in parts)


def _resolve_dir(raw: str, project_root: Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(raw)
    if not path.is_absolute():
        return project_root / path
    return path
