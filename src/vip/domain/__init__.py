"""Domain layer: entities, value objects, protocols, and errors.

Exports
-------
VipError, ConfigError, DataValidationError, PersistenceError, LeakageError
    Platform error types.
RvEstimator, MetricName, PriceFrequency, SplitMode
    Domain enumerations.
Symbol, DateRange, Horizon, ExperimentId
    Immutable value objects.
Instrument, DatasetRef, ExperimentSpec
    Core domain entities.

Notes
-----
Protocols live in ``vip.domain.protocols`` and are imported from there
rather than re-exported here.
"""

from vip.domain.entities import DatasetRef, ExperimentSpec, Instrument
from vip.domain.enums import MetricName, PriceFrequency, RvEstimator, SplitMode
from vip.domain.errors import (
    ConfigError,
    DataValidationError,
    LeakageError,
    PersistenceError,
    VipError,
)
from vip.domain.value_objects import DateRange, ExperimentId, Horizon, Symbol

__all__ = [
    "ConfigError",
    "DataValidationError",
    "DatasetRef",
    "DateRange",
    "ExperimentId",
    "ExperimentSpec",
    "Horizon",
    "Instrument",
    "LeakageError",
    "MetricName",
    "PersistenceError",
    "PriceFrequency",
    "RvEstimator",
    "SplitMode",
    "Symbol",
    "VipError",
]