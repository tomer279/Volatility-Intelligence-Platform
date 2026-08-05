"""Composite use-case: ingest, build features, and screen for N symbols.

Orchestrates existing use-cases in sequence without duplicating
CV, importance, or persistence logic.

Exports
-------
RunStudyConfig
    Settings controlling which pipeline steps to run.
RunStudyStores
    Bundled persistence and data-source dependencies.
run_study
    Execute the full ingest → features → screen pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from vip.application.build_feature_matrix import (
    FeatureMatrixExtras,
    build_and_persist_feature_matrix,
)
from vip.application.ingest_market_data import ingest_market_data
from vip.application.screen_batch import (
    BatchScreenConfig,
    BatchScreenResult,
    run_screen_batch,
)
from vip.application.screen_factors import ScreenConfig
from vip.domain.errors import DataValidationError, PersistenceError
from vip.domain.protocols import MarketDataSource
from vip.domain.value_objects import DateRange, Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_DAYS = 5
VIX_SYMBOL = Symbol("VIX")


@dataclass(frozen=True, slots=True)
class RunStudyConfig:
    """Settings for a composite study run.

    Parameters
    ----------
    symbols : list[Symbol]
        Instruments to process.
    date_range : DateRange
        Ingest window (ignored when ``skip_ingest`` is True).
    horizon_days : int
        Forward target horizon in trading days.
    extras : FeatureMatrixExtras
        Feature-build options (VIX join, jump family, optional family subset).
    skip_ingest : bool
        When True, skip ingestion and require cached market data.
    skip_features : bool
        When True, skip feature building and require cached matrices.
    screen_config : ScreenConfig
        Walk-forward and importance settings forwarded to screening.

    Methods
    -------
    validate()
        Raise ``DataValidationError`` if settings are invalid.
    describe()
        Return a short human-readable summary.
    """

    symbols: list[Symbol]
    date_range: DateRange
    horizon_days: int = DEFAULT_HORIZON_DAYS
    extras: FeatureMatrixExtras = field(default_factory=FeatureMatrixExtras)
    skip_ingest: bool = False
    skip_features: bool = False
    screen_config: ScreenConfig = field(default_factory=ScreenConfig)

    def validate(self) -> None:
        """Raise ``DataValidationError`` if settings are invalid."""
        if not self.symbols:
            raise DataValidationError("symbols must be a non-empty list.")
        if self.horizon_days < 1:
            raise DataValidationError("horizon_days must be at least 1.")
        self.screen_config.validate()

    def describe(self) -> str:
        """Return a short human-readable summary."""
        tickers = ",".join(s.value for s in self.symbols)
        return (
            f"symbols={tickers}, {self.extras.describe()}, "
            f"skip_ingest={self.skip_ingest}, "
            f"skip_features={self.skip_features}"
        )


@dataclass(frozen=True, slots=True)
class RunStudyStores:
    """Bundled persistence and data-source dependencies.

    Parameters
    ----------
    source : MarketDataSource
        External market-data adapter.
    market_store : ParquetMarketDataStore
        Storage for normalized OHLCV.
    feature_store : ParquetFeatureMatrixStore
        Storage for feature matrices.
    artifact_store : FilesystemArtifactStore
        Storage for experiment artifacts.

    Methods
    -------
    has_market_data(symbol)
        Check whether OHLCV exists for ``symbol``.
    has_feature_matrix(symbol)
        Check whether a feature matrix exists for ``symbol``.
    """

    source: MarketDataSource
    market_store: ParquetMarketDataStore
    feature_store: ParquetFeatureMatrixStore
    artifact_store: FilesystemArtifactStore

    def has_market_data(self, symbol: Symbol) -> bool:
        """Check whether OHLCV exists for ``symbol``.

        Parameters
        ----------
        symbol : Symbol
            Instrument to look up.

        Returns
        -------
        bool
            True if the market store contains data for ``symbol``.
        """
        return self.market_store.exists(symbol)

    def has_feature_matrix(self, symbol: Symbol) -> bool:
        """Check whether a feature matrix exists for ``symbol``.

        Parameters
        ----------
        symbol : Symbol
            Instrument to look up.

        Returns
        -------
        bool
            True if the feature store contains a matrix for ``symbol``.
        """
        return self.feature_store.exists(symbol)


def run_study(
        stores: RunStudyStores,
        config: RunStudyConfig,
) -> BatchScreenResult:
    """Execute the full ingest → features → screen pipeline.

    For each symbol the steps are:

    1. **Ingest** OHLCV (skipped when ``skip_ingest`` is True).
    2. **Ingest VIX** when ``with_vix`` is True (same skip rules).
    3. **Build features** including optional VIX columns
       (skipped when ``skip_features`` is True).
    4. **Screen** factors via the existing batch-screening path.

    Single-symbol runs delegate to ``screen_factors`` directly;
    multi-symbol runs delegate to ``run_screen_batch``.

    Parameters
    ----------
    stores : RunStudyStores
        Persistence and data-source dependencies.
    config : RunStudyConfig
        Study settings.

    Returns
    -------
    BatchScreenResult
        Summary table and per-symbol experiment identifiers.

    Raises
    ------
    DataValidationError
        If configuration is invalid.
    PersistenceError
        If required cached data is missing when skip flags are set.
    """
    config.validate()

    _handle_ingestion(stores, config)
    _handle_feature_builds(stores, config)

    return _run_screening(stores, config)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _handle_ingestion(
        stores: RunStudyStores,
        config: RunStudyConfig,
) -> None:
    """Ingest OHLCV for each symbol (and VIX when requested).

    Parameters
    ----------
    stores : RunStudyStores
        Persistence and data-source dependencies.
    config : RunStudyConfig
        Study settings controlling skip behaviour and VIX flag.

    Raises
    ------
    PersistenceError
        If ``skip_ingest`` is True and market data is missing.
    """
    if config.skip_ingest:
        _assert_market_data_exists(
            stores, config.symbols, config.extras.include_vix,
        )
        return

    for symbol in config.symbols:
        if not stores.has_market_data(symbol):
            logger.info("Ingesting %s", symbol.value)
            ingest_market_data(
                source=stores.source,
                store=stores.market_store,
                symbol=symbol,
                date_range=config.date_range,
            )

    if config.extras.include_vix and not stores.has_market_data(VIX_SYMBOL):
        logger.info("Ingesting %s (cross-asset)", VIX_SYMBOL.value)
        ingest_market_data(
            source=stores.source,
            store=stores.market_store,
            symbol=VIX_SYMBOL,
            date_range=config.date_range,
        )


def _assert_market_data_exists(
        stores: RunStudyStores,
        symbols: list[Symbol],
        with_vix: bool,
) -> None:
    """Raise if any required market data is missing.

    Parameters
    ----------
    stores : RunStudyStores
        Persistence dependencies.
    symbols : list[Symbol]
        Required instrument symbols.
    with_vix : bool
        When True, also require VIX data.

    Raises
    ------
    PersistenceError
        If data is missing for any required symbol.
    """
    required = list(symbols)
    if with_vix:
        required.append(VIX_SYMBOL)
    for symbol in required:
        if not stores.has_market_data(symbol):
            raise PersistenceError(
                f"No market data for {symbol.value} and --skip-ingest is set."
            )


def _handle_feature_builds(
        stores: RunStudyStores,
        config: RunStudyConfig,
) -> None:
    """Build feature matrices for each symbol.

    Parameters
    ----------
    stores : RunStudyStores
        Persistence dependencies.
    config : RunStudyConfig
        Study settings controlling skip behaviour and VIX flag.

    Raises
    ------
    PersistenceError
        If ``skip_features`` is True and a feature matrix is missing.
    """
    if config.skip_features:
        _assert_features_exist(stores, config.symbols)
        return

    extras = config.extras
    for symbol in config.symbols:
        if not stores.has_feature_matrix(symbol):
            logger.info("Building features for %s", symbol.value)
            build_and_persist_feature_matrix(
                market_store=stores.market_store,
                feature_store=stores.feature_store,
                symbol=symbol,
                horizon_days=config.horizon_days,
                extras=extras,
            )


def _assert_features_exist(
        stores: RunStudyStores,
        symbols: list[Symbol],
) -> None:
    """Raise if any feature matrix is missing.

    Parameters
    ----------
    stores : RunStudyStores
        Persistence dependencies.
    symbols : list[Symbol]
        Required instrument symbols.

    Raises
    ------
    PersistenceError
        If a feature matrix is missing for any symbol.
    """
    for symbol in symbols:
        if not stores.has_feature_matrix(symbol):
            raise PersistenceError(
                f"No feature matrix for {symbol.value} "
                "and --skip-features is set."
            )


def _run_screening(
        stores: RunStudyStores,
        config: RunStudyConfig,
) -> BatchScreenResult:
    """Delegate to single-symbol or batch screening.

    Parameters
    ----------
    stores : RunStudyStores
        Persistence dependencies.
    config : RunStudyConfig
        Study settings.

    Returns
    -------
    BatchScreenResult
        Summary table and per-symbol experiment identifiers.
    """
    batch_cfg = BatchScreenConfig(
        symbols=config.symbols,
        skip_ingest=True,
        skip_features=True,
        date_range=config.date_range,
        horizon_days=config.horizon_days,
        screen_config=config.screen_config,
    )
    logger.info("Screening %d symbol(s)", len(config.symbols))
    return run_screen_batch(
        source=stores.source,
        market_store=stores.market_store,
        feature_store=stores.feature_store,
        artifact_store=stores.artifact_store,
        config=batch_cfg,
    )
