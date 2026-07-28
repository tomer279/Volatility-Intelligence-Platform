"""Application-layer use-cases.

Exports
-------
IngestMarketDataResult
    Summary of a market data ingestion run.
ingest_market_data
    Fetch, validate, and persist market data.
BuildFeatureMatrixResult
    Summary of a feature-matrix build.
build_and_persist_feature_matrix
    Load OHLCV, build features/target, and persist the matrix.
"""

from vip.application.build_feature_matrix import (
    BuildFeatureMatrixResult,
    build_and_persist_feature_matrix,
)
from vip.application.ingest_market_data import (
    IngestMarketDataResult,
    ingest_market_data,
)

__all__ = [
    "BuildFeatureMatrixResult",
    "IngestMarketDataResult",
    "build_and_persist_feature_matrix",
    "ingest_market_data",
]