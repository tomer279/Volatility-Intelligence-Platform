"""Domain and platform error types.

Exports
-------
VipError
    Base error for all VIP failures.
ConfigError
    Invalid or missing configuration.
DataValidationError
    Invalid market or feature data.
PersistenceError
    Read/write or storage failures.
LeakageError
    Temporal leakage detected in features or splits.
"""


class VipError(Exception):
    """Base error for Volatility Intelligence Platform failures."""


class ConfigError(VipError):
    """Raised when configuration is missing, unreadable, or invalid."""


class DataValidationError(VipError):
    """Raised when market or feature data fails validation checks."""


class PersistenceError(VipError):
    """Raised when loading or saving data/artifacts fails."""


class LeakageError(VipError):
    """Raised when temporal leakage is detected in features or splits."""