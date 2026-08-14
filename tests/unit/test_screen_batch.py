"""Tests for multi-symbol screening batch orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vip.application.screen_batch import BatchScreenConfig, run_screen_batch
from vip.application.screen_factors import ScreenConfig
from vip.domain.value_objects import DateRange, Symbol
from vip.domain.errors import PersistenceError
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore


N_ROWS = 120


def _synthetic_matrix() -> pd.DataFrame:
    """Build a synthetic processed feature matrix (offline)."""
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    rng = np.random.default_rng(7)

    features = pd.DataFrame(
        {
            "rv_cc_1d": rng.uniform(0.01, 0.05, N_ROWS),
            "rv_cc_5d": rng.uniform(0.02, 0.06, N_ROWS),
            "rv_cc_21d": rng.uniform(0.03, 0.07, N_ROWS),
            "ret_1d": rng.normal(0.0, 0.01, N_ROWS),
            "ret_5d": rng.normal(0.0, 0.02, N_ROWS),
            "range_1d": rng.uniform(0.005, 0.02, N_ROWS),
            "range_5d_mean": rng.uniform(0.005, 0.02, N_ROWS),
            "volume_z_21d": rng.normal(0.0, 1.0, N_ROWS),
        },
        index=index,
    )
    target = (
        0.05
        + 0.5 * features["rv_cc_1d"]
        + 0.3 * features["rv_cc_5d"]
        + 0.2 * features["rv_cc_21d"]
        + rng.normal(0.0, 0.001, N_ROWS)
    )
    features["target_rv_cc_5d"] = target
    return features

def _synthetic_ohlcv(n_rows: int = 240) -> pd.DataFrame:
    """Build synthetic canonical OHLCV for rebuild tests."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    close = pd.Series(np.linspace(100.0, 140.0, n_rows), index=index)
    return pd.DataFrame(
        {
            "open": close.to_numpy(),
            "high": (close + 1.0).to_numpy(),
            "low": (close - 1.0).to_numpy(),
            "close": close.to_numpy(),
            "volume": np.linspace(1_000.0, 3_000.0, n_rows),
        },
        index=index,
    )


def _stale_21d_matrix(n_rows: int = 240) -> pd.DataFrame:
    """Matrix that only has ``target_rv_cc_21d``."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    return pd.DataFrame(
        {
            "rv_cc_1d": np.linspace(0.01, 0.05, n_rows),
            "rv_cc_5d": np.linspace(0.02, 0.06, n_rows),
            "rv_cc_21d": np.linspace(0.03, 0.07, n_rows),
            "ret_1d": np.zeros(n_rows),
            "ret_5d": np.zeros(n_rows),
            "range_1d": np.linspace(0.005, 0.02, n_rows),
            "range_5d_mean": np.linspace(0.005, 0.02, n_rows),
            "volume_z_21d": np.zeros(n_rows),
            "target_rv_cc_21d": np.linspace(0.04, 0.08, n_rows),
        },
        index=index,
    )


@dataclass(frozen=True, slots=True)
class _NoOpMarketDataSource:
    """MarketDataSource stub that fails if ingestion is called."""

    def fetch(self, symbol: Symbol, date_range: DateRange) -> pd.DataFrame:
        """Fetch OHLCV (should not be called in this test)."""
        raise AssertionError("Ingestion should be skipped in this test.")

    def source_name(self) -> str:
        """Return a stable source name."""
        return "noop"


def test_screen_batch_skip_ingest_and_features(tmp_path: Path) -> None:
    """Batch orchestration should screen multiple symbols using existing caches."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"

    market_store = ParquetMarketDataStore(root_dir=raw_dir)
    feature_store = ParquetFeatureMatrixStore(root_dir=processed_dir)
    artifact_store = FilesystemArtifactStore(root_dir=artifacts_dir)

    symbols = [Symbol("SPY"), Symbol("QQQ"), Symbol("IWM")]
    for sym in symbols:
        feature_store.save(sym, _synthetic_matrix())

    date_range = DateRange(start=date(2020, 1, 1), end=date(2020, 2, 1))

    screen_cfg = ScreenConfig(
        n_splits=2,
        embargo_size=5,
        n_repeats=1,
        top_k=1,
        random_seed=0,
    )
    batch_cfg = BatchScreenConfig(
        symbols=symbols,
        skip_ingest=True,
        skip_features=True,
        date_range=date_range,
        horizon_days=5,
        screen_config=screen_cfg,
    )

    result = run_screen_batch(
        source=_NoOpMarketDataSource(),
        market_store=market_store,
        feature_store=feature_store,
        artifact_store=artifact_store,
        config=batch_cfg,
    )

    assert set(result.summary["symbol"]) == {s.value for s in symbols}
    assert set(result.experiments.keys()) == set(symbols)

    for sym in symbols:
        exp_id = result.experiments[sym]
        exp_dir = artifact_store.experiment_dir(exp_id)
        assert (exp_dir / "report.html").is_file()
        assert (exp_dir / "metrics_by_regime.json").is_file()


def test_screen_batch_rebuilds_stale_wrong_horizon(tmp_path: Path) -> None:
    """Batch rebuilds when skip_features is False and target horizon mismatches."""
    market_store = ParquetMarketDataStore(tmp_path / "raw")
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    symbol = Symbol("SPY")
    market_store.save(symbol, _synthetic_ohlcv())
    feature_store.save(symbol, _stale_21d_matrix())
    batch_cfg = BatchScreenConfig(
        symbols=[symbol],
        skip_ingest=True,
        skip_features=False,
        date_range=DateRange(start=date(2024, 1, 2), end=date(2024, 4, 1)),
        horizon_days=5,
        screen_config=ScreenConfig(
            n_splits=2,
            embargo_size=5,
            n_repeats=1,
            top_k=1,
            random_seed=0,
        ),
    )
    run_screen_batch(
        source=_NoOpMarketDataSource(),
        market_store=market_store,
        feature_store=feature_store,
        artifact_store=artifact_store,
        config=batch_cfg,
    )
    assert "target_rv_cc_5d" in feature_store.load(symbol).columns


def test_screen_batch_skip_features_wrong_target_raises(
        tmp_path: Path,
) -> None:
    """Batch skip_features fails early on wrong target column."""
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    symbol = Symbol("SPY")
    feature_store.save(symbol, _stale_21d_matrix())
    batch_cfg = BatchScreenConfig(
        symbols=[symbol],
        skip_ingest=True,
        skip_features=True,
        date_range=DateRange(start=date(2024, 1, 2), end=date(2024, 4, 1)),
        horizon_days=5,
        screen_config=ScreenConfig(
            n_splits=2,
            embargo_size=5,
            n_repeats=1,
            top_k=1,
            random_seed=0,
        ),
    )
    with pytest.raises(PersistenceError, match="target_rv_cc_5d"):
        run_screen_batch(
            source=_NoOpMarketDataSource(),
            market_store=ParquetMarketDataStore(tmp_path / "raw"),
            feature_store=feature_store,
            artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
            config=batch_cfg,
        )

