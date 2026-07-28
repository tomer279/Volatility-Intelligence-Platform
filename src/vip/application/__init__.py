"""Application-layer use-cases.

Exports
-------
IngestMarketDataResult
    Summary of a market data ingestion run.
ingest_market_data
    Fetch, validate, and persist market data.
"""

from vip.application.ingest_market_data import (
    IngestMarketDataResult,
    ingest_market_data,
)

__all__ = [
    "IngestMarketDataResult",
    "ingest_market_data",
]