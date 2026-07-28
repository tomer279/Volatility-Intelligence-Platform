"""Application use-case for market data ingestion.

Exports
-------
IngestMarketDataResult
    Summary of a completed ingestion run.
ingest_market_data
    Fetch, validate, and persist daily OHLCV market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vip.domain.protocols import MarketDataSource
from vip.domain.value_objects import DateRange, Symbol
from vip.persistence.parquet_store import ParquetMarketDataStore


@dataclass(frozen=True, slots=True)
class IngestMarketDataResult:
    """Summary of a completed market data ingestion run.

    Parameters
    ----------
    symbol : Symbol
        Ingested instrument.
    row_count : int
        Number of rows written.
    output_path : pathlib.Path
        Destination Parquet file path.
    start_date : str
        Minimum session date in the stored frame.
    end_date : str
        Maximum session date in the stored frame.

    Methods
    -------
    date_span_label()
        Return an inclusive date-span label.
    """

    symbol: Symbol
    row_count: int
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


def ingest_market_data(
    source: MarketDataSource,
    store: ParquetMarketDataStore,
    symbol: Symbol,
    date_range: DateRange,
) -> IngestMarketDataResult:
    """Fetch, validate, and persist daily OHLCV market data.

    Parameters
    ----------
    source : MarketDataSource
        External data source adapter.
    store : ParquetMarketDataStore
        Filesystem-backed Parquet store.
    symbol : Symbol
        Instrument ticker to ingest.
    date_range : DateRange
        Inclusive fetch window.

    Returns
    -------
    IngestMarketDataResult
        Summary of persisted output.
    """
    frame = source.fetch(symbol, date_range)
    output_path = store.save(symbol, frame)

    min_index = frame.index.min()
    max_index = frame.index.max()

    return IngestMarketDataResult(
        symbol=symbol,
        row_count=int(len(frame)),
        output_path=output_path,
        start_date=min_index.date().isoformat(),
        end_date=max_index.date().isoformat(),
    )
