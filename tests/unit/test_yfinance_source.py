"""Tests for Yahoo Finance market data adapter."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.domain.value_objects import DateRange, Symbol
from vip.ingestion.yfinance_source import YFinanceMarketDataSource


def _vendor_like_frame() -> pd.DataFrame:
    """Build a minimal valid vendor-style OHLCV frame.

    Returns
    -------
    pandas.DataFrame
        Daily OHLCV frame using title-cased vendor columns.
    """
    index = pd.to_datetime(
        [
            "2024-01-02 15:30:00+00:00",
            "2024-01-03 15:30:00+00:00",
            "2024-01-04 15:30:00+00:00",
        ]
    )
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1_000, 1_200, 1_500],
        },
        index=index,
    )


def test_source_name_is_stable() -> None:
    """Adapter should expose a stable source identifier."""
    source = YFinanceMarketDataSource()
    assert source.source_name() == "yfinance"


def test_fetch_downloads_and_returns_canonical_frame(mocker: pytest.MockFixture) -> None:
    """Fetch should call yfinance and return normalized canonical OHLCV."""
    source = YFinanceMarketDataSource()
    symbol = Symbol("SPY")
    date_range = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 4))
    raw_frame = _vendor_like_frame()

    download_mock = mocker.patch(
        "vip.ingestion.yfinance_source.yf.download",
        return_value=raw_frame,
    )

    result = source.fetch(symbol, date_range)

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is None
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
    assert result.loc[pd.Timestamp("2024-01-03"), "close"] == pytest.approx(101.5)

    download_mock.assert_called_once_with(
        "SPY",
        start="2024-01-02",
        end="2024-01-05",
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        multi_level_index=False,
    )


def test_fetch_raises_on_empty_download(mocker: pytest.MockFixture) -> None:
    """Fetch should raise when yfinance returns an empty frame."""
    source = YFinanceMarketDataSource()
    symbol = Symbol("SPY")
    date_range = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 4))

    mocker.patch(
        "vip.ingestion.yfinance_source.yf.download",
        return_value=pd.DataFrame(),
    )

    with pytest.raises(DataValidationError, match="No market data returned"):
        source.fetch(symbol, date_range)


def test_fetch_raises_on_download_error(mocker: pytest.MockFixture) -> None:
    """Fetch should wrap yfinance errors as DataValidationError."""
    source = YFinanceMarketDataSource()
    symbol = Symbol("SPY")
    date_range = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 4))

    mocker.patch(
        "vip.ingestion.yfinance_source.yf.download",
        side_effect=RuntimeError("network failure"),
    )

    with pytest.raises(
        DataValidationError,
        match="Failed to download market data from Yahoo Finance",
    ):
        source.fetch(symbol, date_range)


def test_fetch_raises_when_downloaded_data_is_invalid(
    mocker: pytest.MockFixture,
) -> None:
    """Fetch should raise when downloaded frame fails canonical validation."""
    source = YFinanceMarketDataSource()
    symbol = Symbol("SPY")
    date_range = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 4))

    invalid_frame = _vendor_like_frame().drop(columns=["Volume"])
    mocker.patch(
        "vip.ingestion.yfinance_source.yf.download",
        return_value=invalid_frame,
    )

    with pytest.raises(DataValidationError, match="Missing required OHLCV columns"):
        source.fetch(symbol, date_range)
