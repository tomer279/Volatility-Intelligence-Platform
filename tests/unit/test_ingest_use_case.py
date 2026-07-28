"""Tests for market data ingestion use-case."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from vip.application.ingest_market_data import ingest_market_data
from vip.domain.value_objects import DateRange, Symbol
from vip.persistence.parquet_store import ParquetMarketDataStore


class _FakeSource:
    """Simple fake data source for use-case tests."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.called_with: tuple[Symbol, DateRange] | None = None

    def fetch(self, symbol: Symbol, date_range: DateRange) -> pd.DataFrame:
        """Return predefined frame and capture call arguments."""
        self.called_with = (symbol, date_range)
        return self._frame

    def source_name(self) -> str:
        """Return stable fake source name."""
        return "fake-source"


def _canonical_frame() -> pd.DataFrame:
    """Build a canonical OHLCV frame for testing."""
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000, 1_200, 1_500],
        },
        index=index,
    )


def test_ingest_market_data_orchestrates_fetch_and_save(tmp_path: Path) -> None:
    """Use-case should fetch from source, save to store, and return summary."""
    frame = _canonical_frame()
    source = _FakeSource(frame)
    store = ParquetMarketDataStore(tmp_path)

    symbol = Symbol("SPY")
    date_range = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 4))

    result = ingest_market_data(
        source=source,
        store=store,
        symbol=symbol,
        date_range=date_range,
    )

    assert source.called_with == (symbol, date_range)
    assert result.symbol == symbol
    assert result.row_count == 3
    assert result.start_date == "2024-01-02"
    assert result.end_date == "2024-01-04"
    assert result.output_path.is_file()
    assert result.date_span_label() == "2024-01-02..2024-01-04"

    loaded = store.load(symbol)
    pd.testing.assert_frame_equal(loaded, frame)