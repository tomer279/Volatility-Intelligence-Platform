"""Market data ingestion adapters and validators.

Exports
-------
YFinanceMarketDataSource
    Yahoo Finance daily OHLCV adapter.
validate_and_normalize_ohlcv
    Canonical OHLCV validation and normalization entrypoint.
"""

from vip.ingestion.validators import validate_and_normalize_ohlcv
from vip.ingestion.yfinance_source import YFinanceMarketDataSource

__all__ = [
    "YFinanceMarketDataSource",
    "validate_and_normalize_ohlcv",
]