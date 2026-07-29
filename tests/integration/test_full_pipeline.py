"""Integration test: full pipeline with golden-file metric assertions.

Tests
-----
test_full_pipeline_golden_metrics
    Run ingest → features → screen on mock data and assert output metrics
    match the committed golden file within relative tolerance 1e-6.
test_mock_source_produces_valid_features
    Sanity-check that the mock source survives validation and feature building.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from vip.application.build_feature_matrix import build_and_persist_feature_matrix
from vip.application.ingest_market_data import ingest_market_data
from vip.application.screen_factors import ScreenConfig, screen_factors
from vip.domain.value_objects import DateRange, Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore
from vip.domain.protocols import MarketDataSource

GOLDEN_FILE = Path(__file__).parent.parent / "fixtures" / "golden_metrics_spy.json"
METRIC_KEYS = ("qlike", "mse", "mae")
RELATIVE_TOLERANCE = 1e-6
TEST_SYMBOL = Symbol("SPY")
TEST_DATE_RANGE = DateRange(start=date(2020, 1, 1), end=date(2025, 1, 1))
SCREEN_N_SPLITS = 2
SCREEN_EMBARGO = 5


def _run_pipeline(
    mock_source: MarketDataSource,
    market_store: ParquetMarketDataStore,
    feature_store: ParquetFeatureMatrixStore,
    artifact_store: FilesystemArtifactStore,
) -> list[dict[str, object]]:
    """Run ingest → features → screen and return metrics records.

    Parameters
    ----------
    mock_source : MockMarketDataSource
        Deterministic data source.
    market_store : ParquetMarketDataStore
        Temporary market data store.
    feature_store : ParquetFeatureMatrixStore
        Temporary feature matrix store.
    artifact_store : FilesystemArtifactStore
        Temporary artifact store.

    Returns
    -------
    list of dict
        Metrics records loaded from the generated ``metrics.json``.
    """
    ingest_market_data(
        source=mock_source,
        store=market_store,
        symbol=TEST_SYMBOL,
        date_range=TEST_DATE_RANGE,
    )
    build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=TEST_SYMBOL,
    )
    config = ScreenConfig(
        n_splits=SCREEN_N_SPLITS,
        embargo_size=SCREEN_EMBARGO,
        n_repeats=1,
        top_k=1,
        random_seed=0,
    )
    result = screen_factors(
        feature_store=feature_store,
        artifact_store=artifact_store,
        symbol=TEST_SYMBOL,
        config=config,
    )
    metrics_path = (
        artifact_store.experiment_dir(result.identity.experiment_id)
        / "metrics.json"
    )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def test_mock_source_produces_valid_features(
    mock_source: MarketDataSource,
    market_store: ParquetMarketDataStore,
    feature_store: ParquetFeatureMatrixStore,
) -> None:
    """Mock OHLCV survives validation and produces a usable feature matrix."""
    ingest_market_data(
        source=mock_source,
        store=market_store,
        symbol=TEST_SYMBOL,
        date_range=TEST_DATE_RANGE,
    )
    result = build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=TEST_SYMBOL,
    )
    assert result.row_count > 100
    assert result.feature_count > 5


def test_full_pipeline_golden_metrics(
    mock_source: MarketDataSource,
    market_store: ParquetMarketDataStore,
    feature_store: ParquetFeatureMatrixStore,
    artifact_store: FilesystemArtifactStore,
) -> None:
    """Pipeline metrics must match the committed golden file within 1e-6."""
    actual = _run_pipeline(mock_source, market_store, feature_store, artifact_store)

    if not GOLDEN_FILE.exists():
        GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_FILE.write_text(
            json.dumps(actual, indent=2), encoding="utf-8",
        )
        pytest.skip("Golden file generated — commit it and re-run.")

    expected = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))

    assert len(actual) == len(expected), (
        f"Row count mismatch: got {len(actual)}, expected {len(expected)}"
    )
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row["model"] == expected_row["model"]
        for key in METRIC_KEYS:
            assert actual_row[key] == pytest.approx(
                expected_row[key], rel=RELATIVE_TOLERANCE,
            ), f"Metric {key} mismatch for model {actual_row['model']}"
