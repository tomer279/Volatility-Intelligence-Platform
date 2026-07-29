"""Validation and normalization utilities for daily OHLCV data.

Exports
-------
validate_and_normalize_ohlcv
    Normalize vendor OHLCV output into the platform canonical schema.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from vip.domain.errors import DataValidationError


REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

_COLUMN_ALIASES: Mapping[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adj_close": "close",
    "volume": "volume",
}


def validate_and_normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a daily OHLCV table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Raw vendor data with a date-like index and OHLCV-like columns.

    Returns
    -------
    pandas.DataFrame
        Canonical OHLCV frame with:
        - columns: ``open, high, low, close, volume``
        - datetime index: tz-naive UTC-normalized dates
        - strictly increasing unique index

    Raises
    ------
    DataValidationError
        If required columns are missing, index is invalid/duplicated,
        values are missing, or price/volume constraints are violated.
    """
    normalized = normalize_ohlcv_columns(frame)
    validate_required_columns(normalized)

    normalized = normalized.loc[:, list(REQUIRED_COLUMNS)].copy()
    normalized.index = _normalize_datetime_index(normalized.index)

    if normalized.index.has_duplicates:
        duplicate_count = int(normalized.index.duplicated(keep=False).sum())
        raise DataValidationError(
            "Duplicate timestamps detected in OHLCV index. "
            f"Duplicate row count: {duplicate_count}."
        )

    if not normalized.index.is_monotonic_increasing:
        normalized = normalized.sort_index()

    _coerce_required_numeric_columns(normalized)
    normalized = normalized.dropna(subset=list(REQUIRED_COLUMNS))
    validate_no_missing_required_values(normalized)
    validate_price_volume_constraints(normalized)

    return normalized


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to a single level.

    Parameters
    ----------
    frame : pandas.DataFrame
        Vendor frame that may have MultiIndex columns.

    Returns
    -------
    pandas.DataFrame
        Frame with single-level columns.
    """
    if isinstance(frame.columns, pd.MultiIndex):
        flattened = frame.copy()
        flattened.columns = [
            str(level_zero) for level_zero in flattened.columns.get_level_values(0)
        ]
        return flattened
    return frame


def _prefer_close_over_adj_close(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop Adj Close when Close is already present.

    Parameters
    ----------
    frame : pandas.DataFrame
        Vendor frame with title-cased or mixed column names.

    Returns
    -------
    pandas.DataFrame
        Frame without redundant Adj Close when Close exists.
    """
    columns_lower = {str(column).strip().lower(): column for column in frame.columns}
    has_close = "close" in columns_lower
    adj_key = next(
        (key for key in ("adj close", "adj_close") if key in columns_lower),
        None,
    )
    if has_close and adj_key is not None:
        return frame.drop(columns=[columns_lower[adj_key]])
    return frame


def _deduplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each column name.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame that may contain duplicate column labels.

    Returns
    -------
    pandas.DataFrame
        Frame with unique column names.
    """
    return frame.loc[:, ~frame.columns.duplicated(keep="first")].copy()


def normalize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize vendor OHLCV column names to canonical lowercase labels."""
    working = _flatten_columns(frame)
    working = _prefer_close_over_adj_close(working)

    renamed_columns: dict[str, str] = {}
    for column in working.columns:
        key = str(column).strip().lower().replace("-", "_")
        key = key.replace("__", "_")
        key = key.replace("_", " ")
        canonical = _COLUMN_ALIASES.get(key, key.replace(" ", "_"))
        renamed_columns[str(column)] = canonical

    renamed = working.rename(columns=renamed_columns)
    return _deduplicate_columns(renamed)


def validate_required_columns(frame: pd.DataFrame) -> None:
    """Validate that all canonical OHLCV columns are present.

    Parameters
    ----------
    frame : pandas.DataFrame
        Candidate OHLCV frame.

    Raises
    ------
    DataValidationError
        If one or more required columns are missing.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise DataValidationError(
            "Missing required OHLCV columns: "
            f"{missing_text}."
        )


def validate_no_missing_required_values(frame: pd.DataFrame) -> None:
    """Validate that required OHLCV columns have no missing values.

    Parameters
    ----------
    frame : pandas.DataFrame
        Canonical OHLCV frame.

    Raises
    ------
    DataValidationError
        If any required column contains null values.
    """
    null_counts = frame.loc[:, list(REQUIRED_COLUMNS)].isna().sum()
    failing = null_counts[null_counts > 0]
    if not failing.empty:
        details = ", ".join(f"{column}={int(count)}" for column, count in failing.items())
        raise DataValidationError(
            "Null values found in required OHLCV columns: "
            f"{details}."
        )


def validate_price_volume_constraints(frame: pd.DataFrame) -> None:
    """Validate basic daily OHLCV value constraints.

    Parameters
    ----------
    frame : pandas.DataFrame
        Canonical OHLCV frame.

    Raises
    ------
    DataValidationError
        If volume is negative or OHLC bounds are internally inconsistent.
    """
    if (frame["volume"] < 0).any():
        negative_count = int((frame["volume"] < 0).sum())
        raise DataValidationError(
            "Volume contains negative values. "
            f"Row count: {negative_count}."
        )

    max_open_close_low = frame.loc[:, ["open", "close", "low"]].max(axis=1)
    min_open_close_high = frame.loc[:, ["open", "close", "high"]].min(axis=1)

    high_fail = frame["high"] < max_open_close_low
    low_fail = frame["low"] > min_open_close_high

    if high_fail.any() or low_fail.any():
        high_fail_count = int(high_fail.sum())
        low_fail_count = int(low_fail.sum())
        raise DataValidationError(
            "Price bounds violated in OHLC data. "
            f"Rows with high below max(open, close, low): {high_fail_count}; "
            f"rows with low above min(open, close, high): {low_fail_count}."
        )


def _normalize_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """Convert index to UTC-normalized, tz-naive ``DatetimeIndex``.

    Parameters
    ----------
    index : pandas.Index
        Candidate index from vendor output.

    Returns
    -------
    pandas.DatetimeIndex
        UTC-normalized, tz-naive daily index.

    Raises
    ------
    DataValidationError
        If conversion to datetime index fails.
    """
    try:
        datetime_index = pd.to_datetime(index, errors="raise", format="ISO8601")
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Index cannot be converted to datetime.") from exc

    if not isinstance(datetime_index, pd.DatetimeIndex):
        raise DataValidationError("Index is not a DatetimeIndex after conversion.")

    if datetime_index.tz is not None:
        datetime_index = datetime_index.tz_convert("UTC").tz_localize(None)

    return datetime_index.normalize()


def _coerce_required_numeric_columns(frame: pd.DataFrame) -> None:
    """Coerce required OHLCV columns to numeric dtype in-place.

    Parameters
    ----------
    frame : pandas.DataFrame
        Canonical OHLCV frame candidate.

    Raises
    ------
    DataValidationError
        If numeric coercion fails for any required column.
    """
    for column in REQUIRED_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            series_or_frame = frame[column]
            if isinstance(series_or_frame, pd.DataFrame):
                raise DataValidationError(
                    f"Duplicate or multi-dimensional values found for column: {column}."
                )
            frame[column] = pd.to_numeric(series_or_frame, errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                "Non-numeric values found in OHLCV column: "
                f"{column}."
            ) from exc
