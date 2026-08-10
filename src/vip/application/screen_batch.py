"""Multi-symbol screening orchestration.

Exports
-------
BatchScreenConfig
    Configuration for multi-symbol screening runs.
BatchScreenResult
    Summary table plus per-symbol experiment identifiers.
run_screen_batch
    Loop over symbols and run ingest/features/screen; rebuild features unless skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pandas as pd

from vip.application.build_feature_matrix import (
    build_and_persist_feature_matrix,
    require_cached_feature_target,
)
from vip.application.ingest_market_data import ingest_market_data
from vip.application.screen_factors import (
    ScreenConfig,
    screen_factors,
    settings_for_horizon,
)
from vip.domain.errors import DataValidationError, PersistenceError
from vip.domain.protocols import MarketDataSource
from vip.domain.value_objects import DateRange, ExperimentId, Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore


@dataclass(frozen=True, slots=True)
class BatchScreenConfig:
    """Settings for a multi-symbol screen batch.

    Parameters
    ----------
    symbols : list[Symbol]
        Symbols to screen.
    skip_ingest : bool
        When True, do not ingest missing market data.
    skip_features : bool
        When True, do not rebuild feature matrices; require
        ``target_rv_cc_{horizon_days}d`` in the cached matrix.
    date_range : DateRange
        Ingest window (used only when ingestion is not skipped).
    horizon_days : int
        Forward target horizon used for feature building.
    screen_config : ScreenConfig
        Per-symbol screening configuration.

    Methods
    -------
    validate()
        Raise if configuration is invalid.
    describe()
        Return a short human-readable summary.
    """

    symbols: list[Symbol]
    skip_ingest: bool
    skip_features: bool
    date_range: DateRange
    horizon_days: int
    screen_config: ScreenConfig

    def validate(self) -> None:
        """Raise ``DataValidationError`` when configuration is invalid."""
        if not self.symbols:
            raise DataValidationError("symbols must be a non-empty list.")
        if self.horizon_days < 1:
            raise DataValidationError("horizon_days must be at least 1.")
        if self.date_range.end is not None and self.date_range.end < self.date_range.start:
            raise DataValidationError("date_range end must be >= start.")
        self.screen_config.validate()

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact summary of batch settings.
        """
        return (
            f"symbols={len(self.symbols)}, skip_ingest={self.skip_ingest}, "
            f"skip_features={self.skip_features}, horizon_days={self.horizon_days}"
        )


@dataclass(frozen=True, slots=True)
class BatchScreenResult:
    """Outputs from a multi-symbol screen batch.

    Parameters
    ----------
    summary : pandas.DataFrame
        One row per symbol: best model, best QLIKE, top feature.
    experiments : dict[Symbol, ExperimentId]
        Per-symbol experiment identifiers used by artifacts.
    """

    summary: pd.DataFrame
    experiments: dict[Symbol, ExperimentId]


def run_screen_batch(
        source: MarketDataSource,
        market_store: ParquetMarketDataStore,
        feature_store: ParquetFeatureMatrixStore,
        artifact_store: FilesystemArtifactStore,
        config: BatchScreenConfig,
) -> BatchScreenResult:
    """Run ingest/features/screen for multiple symbols with caching.

    Parameters
    ----------
    source : vip.domain.protocols.MarketDataSource
        Market data adapter (used only when ingestion is not skipped).
    market_store : ParquetMarketDataStore
        Storage for normalized OHLCV.
    feature_store : ParquetFeatureMatrixStore
        Storage for persisted feature matrices.
    artifact_store : FilesystemArtifactStore
        Storage for experiment artifacts.
    config : BatchScreenConfig
        Batch settings.

    Returns
    -------
    BatchScreenResult
        Summary table and per-symbol experiment ids.

    Raises
    ------
    PersistenceError
        If required cached data is missing when skip flags are enabled.
    DataValidationError
        If configuration is invalid.
    """
    config.validate()

    rows: list[dict[str, object]] = []
    experiments: dict[Symbol, ExperimentId] = {}

    for symbol in config.symbols:
        _ensure_caches_ready(
            source, market_store, feature_store, symbol, config,
        )

        horizon_defaults = settings_for_horizon(config.horizon_days)
        result = screen_factors(
            feature_store=feature_store,
            artifact_store=artifact_store,
            symbol=symbol,
            config=replace(
                config.screen_config,
                embargo_size=horizon_defaults.config.embargo_size,
            ),
            inference=horizon_defaults.inference,
        )
        experiments[symbol] = result.identity.experiment_id

        best_row = result.tables.summary.iloc[0]
        rows.append(
            {
                "symbol": symbol.value,
                "best_model": str(best_row["model"]),
                "best_qlike": float(best_row["qlike"]),
                "top_feature": result.top_feature(),
                "experiment_id": result.identity.experiment_id.value,
                "generated_on": date.today().isoformat(),
            }
        )

    summary = pd.DataFrame.from_records(rows)
    return BatchScreenResult(summary=summary, experiments=experiments)


def _ensure_caches_ready(
        source: MarketDataSource,
        market_store: ParquetMarketDataStore,
        feature_store: ParquetFeatureMatrixStore,
        symbol: Symbol,
        config: BatchScreenConfig,
) -> None:
    """Ensure market data and feature caches exist for a symbol.

    Raises
    ------
    PersistenceError
        If a required cache is missing and the corresponding skip flag is set.
    """
    if not config.skip_ingest and not market_store.exists(symbol):
        ingest_market_data(
            source=source,
            store=market_store,
            symbol=symbol,
            date_range=config.date_range,
        )

    if config.skip_features:
        require_cached_feature_target(
            feature_store=feature_store,
            symbol=symbol,
            horizon_days=config.horizon_days,
        )
        return

    if config.skip_ingest and not market_store.exists(symbol):
        raise PersistenceError(
            f"Missing market data for {symbol.value}, "
            "but --skip-ingest was set."
        )
    build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=symbol,
        horizon_days=config.horizon_days,
    )
