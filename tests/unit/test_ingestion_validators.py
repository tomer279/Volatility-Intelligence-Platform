"""Tests for OHLCV ingestion validation and normalization."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.ingestion.validators import validate_and_normalize_ohlcv


def _valid_vendor_frame() -> pd.DataFrame:
    """Build a minimal valid vendor-like OHLCV frame.

    Returns
    -------
    pandas.DataFrame
        Valid daily OHLCV frame with title-cased vendor columns.
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


def test_validate_and_normalize_returns_canonical_schema() -> None:
    """Valid vendor-like data should normalize to canonical OHLCV output."""
    frame = _valid_vendor_frame()

    result = validate_and_normalize_ohlcv(frame)

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is None
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
    assert result.index[0] == pd.Timestamp("2024-01-02")
    assert result.loc[pd.Timestamp("2024-01-03"), "close"] == pytest.approx(101.5)


def test_missing_required_column_raises() -> None:
    """Missing required OHLCV column should raise DataValidationError."""
    frame = _valid_vendor_frame().drop(columns=["Volume"])

    with pytest.raises(DataValidationError, match="Missing required OHLCV columns"):
        validate_and_normalize_ohlcv(frame)


def test_duplicate_index_raises() -> None:
    """Duplicate timestamps in index should raise DataValidationError."""
    frame = _valid_vendor_frame()
    duplicated_index = pd.to_datetime(
        [
            "2024-01-02 15:30:00+00:00",
            "2024-01-02 15:31:00+00:00",
            "2024-01-03 15:30:00+00:00",
        ]
    )
    frame.index = duplicated_index

    with pytest.raises(DataValidationError, match="Duplicate timestamps"):
        validate_and_normalize_ohlcv(frame)


def test_unsorted_index_is_sorted() -> None:
    """Unsorted index should be sorted ascending by timestamp."""
    frame = _valid_vendor_frame().iloc[[2, 0, 1]]

    result = validate_and_normalize_ohlcv(frame)

    expected_index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    pd.testing.assert_index_equal(result.index, expected_index)


def test_negative_volume_raises() -> None:
    """Negative volume should raise DataValidationError."""
    frame = _valid_vendor_frame()
    frame.loc[frame.index[1], "Volume"] = -10

    with pytest.raises(DataValidationError, match="Volume contains negative values"):
        validate_and_normalize_ohlcv(frame)


def test_price_bounds_violation_raises() -> None:
    """Inconsistent high/low bounds should raise DataValidationError."""
    frame = _valid_vendor_frame()
    frame.loc[frame.index[0], "High"] = 98.0

    with pytest.raises(DataValidationError, match="Price bounds violated"):
        validate_and_normalize_ohlcv(frame)


def test_adj_close_alias_maps_to_close() -> None:
    """Vendor Adj Close should normalize into canonical close column."""
    frame = _valid_vendor_frame().drop(columns=["Close"])
    frame["Adj Close"] = [100.6, 101.6, 102.6]

    result = validate_and_normalize_ohlcv(frame)

    assert "close" in result.columns
    assert result.loc[pd.Timestamp("2024-01-03"), "close"] == pytest.approx(101.6)


def test_non_datetime_index_raises() -> None:
    """Index that cannot be converted to datetime should raise DataValidationError."""
    frame = _valid_vendor_frame().copy()
    frame.index = pd.Index(["2024-13-01", "2024-13-02", "2024-13-03"])

    with pytest.raises(DataValidationError, match="Index cannot be converted to datetime"):
        validate_and_normalize_ohlcv(frame)


def test_timezone_aware_index_becomes_utc_naive() -> None:
    """Timezone-aware timestamps should convert to UTC-normalized naive dates."""
    index = pd.DatetimeIndex(
        [
            datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc),
            datetime(2024, 1, 3, 10, 30, tzinfo=timezone.utc),
            datetime(2024, 1, 4, 10, 30, tzinfo=timezone.utc),
        ]
    )
    frame = _valid_vendor_frame()
    frame.index = index

    result = validate_and_normalize_ohlcv(frame)

    assert result.index.tz is None
    assert result.index[0] == pd.Timestamp("2024-01-02")

def test_close_preferred_when_adj_close_also_present() -> None:
    """When both Close and Adj Close exist, prefer Close."""
    frame = _valid_vendor_frame()
    frame["Adj Close"] = [90.0, 91.0, 92.0]

    result = validate_and_normalize_ohlcv(frame)

    assert result.loc[pd.Timestamp("2024-01-02"), "close"] == pytest.approx(100.5)
