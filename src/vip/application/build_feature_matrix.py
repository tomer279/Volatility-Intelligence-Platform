"""Application use-case for building and persisting feature matrices.

Exports
-------
BuildFeatureMatrixResult
    Summary of a completed feature-matrix build.
build_and_persist_feature_matrix
    Load OHLCV, build features/target, and persist the matrix.
require_cached_feature_target
    Fail fast when a skipped rebuild lacks the expected target column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import Symbol
from vip.features.pipeline import (
    VixJoinOptions,
    build_feature_matrix,
)
from vip.features.registry import create_default_registry
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore

DEFAULT_HORIZON_DAYS = 5


@dataclass(frozen=True, slots=True)
class BuildFeatureMatrixResult:
    """Summary of a completed feature-matrix build.

    Parameters
    ----------
    symbol : Symbol
        Instrument used for the build.
    row_count : int
        Number of rows in the cleaned matrix.
    feature_count : int
        Number of feature columns (excludes target).
    output_path : pathlib.Path
        Destination Parquet path.
    start_date : str
        Minimum session date in the matrix.
    end_date : str
        Maximum session date in the matrix.

    Methods
    -------
    date_span_label()
        Return an inclusive date-span label.
    """

    symbol: Symbol
    row_count: int
    feature_count: int
    output_path: Path
    start_date: str
    end_date: str

    def date_span_label(self) -> str:
        """Return an inclusive date-span label.

        Returns
        -------
        str
            Date span text in ``start..end`` form.
        """
        return f"{self.start_date}..{self.end_date}"


@dataclass(frozen=True, slots=True)
class FeatureMatrixExtras:
    """Optional feature-build settings beyond horizon.

    Parameters
    ----------
    feature_names : list of str or None, default None
        Subset of feature-family names. ``None`` uses all registered families.
    include_vix : bool, default False
        When True, load VIX from the market store and join features.
    vix_symbol : Symbol or None, default None
        Storage symbol for VIX. ``None`` means ``Symbol("VIX")``.
    include_jump : bool, default False
        When True, register the daily ``jump`` feature family.
    include_iv_rv : bool, default False
        When True, append the ``iv_rv`` gap family. Implies loading VIX
        (same as ``include_vix=True``) even if ``include_vix`` is False.
    include_rates : bool, default False
        When True, load TNX from the market store and join yield features.
    rates_symbol : Symbol or None, default None
        Storage symbol for the Treasury yield proxy. ``None`` means
        ``Symbol("TNX")``.

    Methods
    -------
    resolved_vix_symbol()
        Return the VIX storage symbol.
    resolved_rates_symbol()
        Return the rates storage symbol.
    needs_vix()
        Return whether VIX OHLCV must be loaded.
    needs_rates()
        Return whether rates OHLCV must be loaded.
    describe()
        Return a short human-readable summary.
    """
    feature_names: list[str] | None = None
    include_vix: bool = False
    vix_symbol: Symbol | None = None
    include_jump: bool = False
    include_iv_rv: bool = False
    include_rates: bool = False
    rates_symbol: Symbol | None = None

    def resolved_vix_symbol(self) -> Symbol:
        """Return the VIX storage symbol.

        Returns
        -------
        Symbol
            Configured VIX symbol or ``VIX``.
        """
        return self.vix_symbol if self.vix_symbol is not None else Symbol("VIX")

    def resolved_rates_symbol(self) -> Symbol:
        """Return the rates storage symbol.

        Returns
        -------
        Symbol
            Configured rates symbol or ``TNX``.
        """
        if self.rates_symbol is not None:
            return self.rates_symbol
        return Symbol("TNX")

    def needs_vix(self) -> bool:
        """Return whether VIX OHLCV must be loaded.

        Returns
        -------
        bool
            True when ``include_vix`` or ``include_iv_rv`` is set.
        """
        return self.include_vix or self.include_iv_rv

    def needs_rates(self) -> bool:
        """Return whether rates OHLCV must be loaded.

        Returns
        -------
        bool
            True when ``include_rates`` is set.
        """
        return self.include_rates

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact extras summary including family subset and all
            include_* flags. Appends ``(iv_rv implies VIX load)`` when
            gaps are requested without an explicit ``include_vix``.
        """
        names = (
            "all"
            if self.feature_names is None
            else ",".join(self.feature_names)
        )
        summary = (
            f"families={names}, "
            f"include_vix={self.include_vix}, "
            f"include_jump={self.include_jump}, "
            f"include_iv_rv={self.include_iv_rv}, "
            f"include_rates={self.include_rates}"
        )
        if self.include_iv_rv and not self.include_vix:
            return f"{summary} (iv_rv implies VIX load)"
        return summary


def build_and_persist_feature_matrix(
        market_store: ParquetMarketDataStore,
        feature_store: ParquetFeatureMatrixStore,
        symbol: Symbol,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        extras: FeatureMatrixExtras | None = None,
) -> BuildFeatureMatrixResult:
    """Load OHLCV, build a feature matrix, and persist it.

    Parameters
    ----------
    market_store : ParquetMarketDataStore
        Store containing canonical raw OHLCV.
    feature_store : ParquetFeatureMatrixStore
        Destination store for the feature matrix.
    symbol : Symbol
        Instrument to process.
    horizon_days : int, default 5
        Forward target horizon in trading days.
    extras : FeatureMatrixExtras or None, default None
        Optional VIX / jump / IV−RV / rates settings. ``None`` uses defaults.

    Returns
    -------
    BuildFeatureMatrixResult
        Summary of the persisted matrix.

    Raises
    ------
    PersistenceError
        If market data is missing.
    """
    if not market_store.exists(symbol):
        raise PersistenceError(
            f"No market data found for {symbol.value}. Run ingest first."
        )

    resolved = extras if extras is not None else FeatureMatrixExtras()
    ohlcv = market_store.load(symbol)
    cross_asset = _resolve_cross_asset_join(market_store, resolved)
    registry = create_default_registry(include_jump=resolved.include_jump)
    matrix = build_feature_matrix(
        ohlcv,
        horizon_days=horizon_days,
        feature_names=resolved.feature_names,
        registry=registry,
        vix_ohlcv=cross_asset,
    )
    output_path = feature_store.save(symbol, matrix)
    feature_count = _feature_column_count(matrix, horizon_days)

    return BuildFeatureMatrixResult(
        symbol=symbol,
        row_count=int(matrix.shape[0]),
        feature_count=feature_count,
        output_path=output_path,
        start_date=matrix.index.min().date().isoformat(),
        end_date=matrix.index.max().date().isoformat(),
    )


def require_cached_feature_target(
        feature_store: ParquetFeatureMatrixStore,
        symbol: Symbol,
        horizon_days: int,
) -> None:
    """Raise if a cached matrix is missing or lacks the horizon target.

    Parameters
    ----------
    feature_store : ParquetFeatureMatrixStore
        Persistence for feature matrices.
    symbol : Symbol
        Instrument whose matrix must be present.
    horizon_days : int
        Expected forward target horizon (names ``target_rv_cc_{h}d``).

    Raises
    ------
    PersistenceError
        If the matrix file is missing or does not contain the target column.
        Message suggests ``vip features --symbol … --horizon … [--with …]``.
    """
    target_column = f"target_rv_cc_{horizon_days}d"
    features_hint = (
        f"vip features --symbol {symbol.value} "
        f"--horizon {horizon_days} [--with ...]"
    )
    if not feature_store.exists(symbol):
        raise PersistenceError(
            f"No feature matrix for {symbol.value} "
            f"and --skip-features is set. Rebuild with: {features_hint}"
        )
    matrix = feature_store.load(symbol)
    if target_column not in matrix.columns:
        raise PersistenceError(
            f"Feature matrix for {symbol.value} missing target column "
            f"'{target_column}'. Rebuild with: {features_hint}"
        )


def _resolve_cross_asset_join(
        market_store: ParquetMarketDataStore,
        extras: FeatureMatrixExtras,
) -> VixJoinOptions | None:
    """Load optional VIX / rates OHLCV and build pipeline join options.

    Parameters
    ----------
    market_store : ParquetMarketDataStore
        Raw OHLCV store.
    extras : FeatureMatrixExtras
        Flags controlling which auxiliary series to load.

    Returns
    -------
    VixJoinOptions or None
        Join options when any cross-asset series is required; otherwise
        ``None``.

    Raises
    ------
    PersistenceError
        If a required auxiliary symbol is missing from the store.
    """
    if not extras.needs_vix() and not extras.needs_rates():
        return None

    vix_ohlcv = None
    rates_ohlcv = None
    if extras.needs_vix():
        vix_ohlcv = _load_aux_ohlcv(
            market_store,
            extras.resolved_vix_symbol(),
            "Run: vip ingest --symbol VIX",
        )
    if extras.needs_rates():
        rates_ohlcv = _load_aux_ohlcv(
            market_store,
            extras.resolved_rates_symbol(),
            "Run: vip ingest --symbol TNX",
        )
    return VixJoinOptions(
        vix_ohlcv=vix_ohlcv,
        include_iv_rv=extras.include_iv_rv,
        rates_ohlcv=rates_ohlcv,
    )


def _load_aux_ohlcv(
        market_store: ParquetMarketDataStore,
        symbol: Symbol,
        missing_hint: str,
) -> pd.DataFrame:
    """Load one auxiliary OHLCV frame or raise ``PersistenceError``.

    Parameters
    ----------
    market_store : ParquetMarketDataStore
        Raw OHLCV store.
    symbol : Symbol
        Auxiliary storage symbol (for example ``VIX`` or ``TNX``).
    missing_hint : str
        Actionable message appended when the symbol is absent.

    Returns
    -------
    pandas.DataFrame
        Canonical OHLCV for ``symbol``.

    Raises
    ------
    PersistenceError
        If ``symbol`` is not present in the store.
    """
    if not market_store.exists(symbol):
        raise PersistenceError(
            f"No market data found for {symbol.value}. {missing_hint}"
        )
    return market_store.load(symbol)


def _feature_column_count(matrix: pd.DataFrame, horizon_days: int) -> int:
    """Count feature columns excluding the forward-RV target.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Persisted feature matrix.
    horizon_days : int
        Horizon used to name ``target_rv_cc_{h}d``.

    Returns
    -------
    int
        Number of non-target columns.
    """
    target_column = f"target_rv_cc_{horizon_days}d"
    has_target = 1 if target_column in matrix.columns else 0
    return int(matrix.shape[1] - has_target)
