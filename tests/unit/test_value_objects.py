"""Tests for domain value objects."""

from datetime import date

import pytest

from vip.domain.errors import DataValidationError
from vip.domain.value_objects import DateRange, ExperimentId, Horizon, Symbol


def test_symbol_normalizes_ticker() -> None:
    """Symbol strips, uppercases, and exposes path/match helpers."""
    symbol = Symbol(" spy ")
    assert symbol.value == "SPY"
    assert symbol.as_path_key() == "SPY"
    assert symbol.matches("spy")
    assert symbol.matches(" SPY ")
    assert not symbol.matches("QQQ")


def test_symbol_rejects_empty() -> None:
    """Symbol raises when the ticker is empty after stripping."""
    with pytest.raises(DataValidationError):
        Symbol("")
    with pytest.raises(DataValidationError):
        Symbol("   ")


def test_date_range_helpers() -> None:
    """DateRange supports containment and inclusive day span."""
    span = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 3))
    assert span.contains(date(2020, 1, 1))
    assert span.contains(date(2020, 1, 2))
    assert span.contains(date(2020, 1, 3))
    assert not span.contains(date(2019, 12, 31))
    assert not span.contains(date(2020, 1, 4))
    assert span.day_span() == 3


def test_date_range_rejects_inverted_bounds() -> None:
    """DateRange raises when start is after end."""
    with pytest.raises(DataValidationError):
        DateRange(start=date(2020, 2, 1), end=date(2020, 1, 1))


def test_date_range_allows_single_day() -> None:
    """DateRange allows start equal to end."""
    span = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 1))
    assert span.day_span() == 1
    assert span.contains(date(2020, 1, 1))


def test_horizon_helpers_and_validation() -> None:
    """Horizon validates trading_days and exposes helpers."""
    horizon = Horizon(5)
    assert horizon.is_short_term()
    assert horizon.label() == "5d"
    assert not Horizon(6).is_short_term()
    with pytest.raises(DataValidationError):
        Horizon(0)
    with pytest.raises(DataValidationError):
        Horizon(-1)


def test_experiment_id_normalizes_and_helpers() -> None:
    """ExperimentId strips, rejects empty, and supports prefix checks."""
    experiment_id = ExperimentId("  run-1 ")
    assert experiment_id.value == "run-1"
    assert experiment_id.as_path_key() == "run-1"
    assert experiment_id.starts_with("run")
    assert not experiment_id.starts_with("exp")
    with pytest.raises(DataValidationError):
        ExperimentId("")
    with pytest.raises(DataValidationError):
        ExperimentId("   ")