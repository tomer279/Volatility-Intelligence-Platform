"""Application use-case for building and persisting feature matrices.

Exports
-------
BuildFeatureMatrixResult
    Summary of a completed feature-matrix build.
build_and_persist_feature_matrix
    Load OHLCV, build features/target, and persist the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import Symbol
from vip.features.pipeline import build_feature_matrix
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
        Subset of feature-family names.
    include_vix : bool, default False
        When True, load VIX from the market store and join features.
    vix_symbol : Symbol or None, default None
        Storage symbol for VIX. ``None`` means ``Symbol("VIX")``.

    Methods
    -------
    resolved_vix_symbol()
        Return the VIX storage symbol.
    describe()
        Return a short human-readable summary.
    """
    feature_names: list[str] | None = None
    include_vix: bool = False
    vix_symbol: Symbol | None = None

    def resolved_vix_symbol(self) -> Symbol:
        """Return the VIX storage symbol.

        Returns
        -------
        Symbol
            Configured VIX symbol or ``VIX``.
        """
        return self.vix_symbol if self.vix_symbol is not None else Symbol("VIX")

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact extras summary.
        """
        names = "all" if self.feature_names is None else ",".join(self.feature_names)
        return f"families={names}, include_vix={self.include_vix}"


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
    feature_names : list of str or None, default None
        Optional subset of feature-family names.

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

    vix_ohlcv = None
    if resolved.include_vix:
        vix_symbol = resolved.resolved_vix_symbol()
        if not market_store.exists(vix_symbol):
            raise PersistenceError(
                f"No market data found for {vix_symbol.value}. "
                "Run: vip ingest --symbol VIX"
            )
        vix_ohlcv = market_store.load(vix_symbol)

    matrix = build_feature_matrix(
        ohlcv,
        horizon_days=horizon_days,
        feature_names=resolved.feature_names,
        vix_ohlcv=vix_ohlcv,
    )
    output_path = feature_store.save(symbol, matrix)

    target_column = f"target_rv_cc_{horizon_days}d"
    feature_count = int(matrix.shape[1] - (1 if target_column in matrix.columns else 0))

    return BuildFeatureMatrixResult(
        symbol=symbol,
        row_count=int(matrix.shape[0]),
        feature_count=feature_count,
        output_path=output_path,
        start_date=matrix.index.min().date().isoformat(),
        end_date=matrix.index.max().date().isoformat(),
    )
