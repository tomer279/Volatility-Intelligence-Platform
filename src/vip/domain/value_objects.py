"""Immutable domain value objects.

Exports
-------
Symbol
    Normalized ticker identifier.
DateRange
    Inclusive calendar date window.
Horizon
    Forecast horizon in trading days.
ExperimentId
    Opaque experiment identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from vip.domain.errors import DataValidationError


@dataclass(frozen=True, slots=True)
class Symbol:
    """Normalized equity or ETF ticker.

    Parameters
    ----------
    value : str
        Raw ticker string. Stored stripped and uppercased.

    Methods
    -------
    as_path_key()
        Return the ticker form used in storage paths.
    matches(other)
        Return True if ``other`` equals this ticker, ignoring case.
    """

    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip().upper()
        if not normalized:
            raise DataValidationError("Symbol must be a non-empty ticker.")
        object.__setattr__(self, "value", normalized)

    def as_path_key(self) -> str:
        """Return the ticker form used in storage paths.

        Returns
        -------
        str
            Uppercase ticker suitable for directory names.
        """
        return self.value

    def matches(self, other: str) -> bool:
        """Check equality against a raw ticker string.

        Parameters
        ----------
        other : str
            Ticker to compare (case and padding ignored).

        Returns
        -------
        bool
            True if ``other`` refers to the same symbol.
        """
        return self.value == other.strip().upper()


@dataclass(frozen=True, slots=True)
class DateRange:
    """Inclusive calendar date range for data requests.

    Parameters
    ----------
    start : datetime.date
        First day of the range (inclusive).
    end : datetime.date
        Last day of the range (inclusive).

    Methods
    -------
    contains(day)
        Return whether ``day`` lies inside the range.
    day_span()
        Return the inclusive calendar-day count.

    Raises
    ------
    DataValidationError
        If ``start`` is after ``end``.
    """

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise DataValidationError(
                "DateRange start must be on or before end."
            )

    def contains(self, day: date) -> bool:
        """Return whether a date lies inside the range.

        Parameters
        ----------
        day : datetime.date
            Date to test.

        Returns
        -------
        bool
            True if ``start <= day <= end``.
        """
        return self.start <= day <= self.end

    def day_span(self) -> int:
        """Return the inclusive calendar-day count.

        Returns
        -------
        int
            Number of calendar days from ``start`` through ``end``.
        """
        return (self.end - self.start).days + 1


@dataclass(frozen=True, slots=True)
class Horizon:
    """Forecast horizon measured in trading days.

    Parameters
    ----------
    trading_days : int
        Number of trading sessions in the forecast window. Must be >= 1.

    Methods
    -------
    is_short_term()
        Return True when the horizon is at most five sessions.
    label()
        Return a compact label such as ``'5d'``.

    Raises
    ------
    DataValidationError
        If ``trading_days`` is less than 1.
    """

    trading_days: int

    def __post_init__(self) -> None:
        if self.trading_days < 1:
            raise DataValidationError(
                "Horizon trading_days must be at least 1."
            )

    def is_short_term(self) -> bool:
        """Return whether this is a short-term horizon.

        Returns
        -------
        bool
            True when ``trading_days`` is at most 5.
        """
        return self.trading_days <= 5

    def label(self) -> str:
        """Return a compact label for configs and column names.

        Returns
        -------
        str
            Label of the form ``'{trading_days}d'``.
        """
        return f"{self.trading_days}d"


@dataclass(frozen=True, slots=True)
class ExperimentId:
    """Opaque identifier for a reproducible experiment run.

    Parameters
    ----------
    value : str
        Raw identifier. Stored stripped; must be non-empty after strip.

    Methods
    -------
    as_path_key()
        Return the filesystem-safe id string.
    starts_with(prefix)
        Return whether this id begins with ``prefix``.

    Raises
    ------
    DataValidationError
        If ``value`` is empty after stripping.
    """

    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip()
        if not normalized:
            raise DataValidationError("ExperimentId must be non-empty.")
        object.__setattr__(self, "value", normalized)

    def as_path_key(self) -> str:
        """Return the id form used under ``artifacts/``.

        Returns
        -------
        str
            Stripped experiment identifier.
        """
        return self.value

    def starts_with(self, prefix: str) -> bool:
        """Return whether this id begins with a prefix.

        Parameters
        ----------
        prefix : str
            Prefix to test.

        Returns
        -------
        bool
            True if ``value`` starts with ``prefix``.
        """
        return self.value.startswith(prefix)