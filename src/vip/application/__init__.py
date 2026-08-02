"""Application-layer use-cases.

Exports
-------
IngestMarketDataResult
    Summary of a market data ingestion run.
ingest_market_data
    Fetch, validate, and persist market data.
BuildFeatureMatrixResult
    Summary of a feature-matrix build.
FeatureMatrixExtras
    Optional feature-build settings (families, include_vix).
build_and_persist_feature_matrix
    Load OHLCV, build features/target, and persist the matrix.
BaselineExperimentResult
    Summary of a baseline walk-forward experiment.
run_baseline_experiment
    Load features, evaluate baselines, and persist metrics artifacts.
FactorScreenResult
    Summary of a factor-screening experiment.
ScreenConfig
    Walk-forward and importance settings for screening.
screen_factors
    Load features, race models, rank factors, and persist artifacts.
ScreenInferenceOptions
    Bootstrap / HLN / non-overlap sensitivity settings for screening.
"""

from vip.application.build_feature_matrix import (
    BuildFeatureMatrixResult,
    build_and_persist_feature_matrix,
    FeatureMatrixExtras
)
from vip.application.ingest_market_data import (
    IngestMarketDataResult,
    ingest_market_data,
)
from vip.application.run_baseline_experiment import (
    BaselineExperimentResult,
    run_baseline_experiment,
)
from vip.application.screen_factors import (
    FactorScreenResult,
    ScreenConfig,
    ScreenInferenceOptions,
    screen_factors,
)

__all__ = [
    "BaselineExperimentResult",
    "BuildFeatureMatrixResult",
    "IngestMarketDataResult",
    "build_and_persist_feature_matrix",
    "ingest_market_data",
    "run_baseline_experiment",
    "FactorScreenResult",
    "ScreenConfig",
    "screen_factors",
    "FeatureMatrixExtras",
    "ScreenInferenceOptions",
]