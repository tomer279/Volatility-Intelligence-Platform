"""Tests for vip run feature rebuild vs stale shared parquet cache.

Exports
-------
(test module; no public API)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vip.application.build_feature_matrix import FeatureMatrixExtras
from vip.application.run_study import (
    RunStudyConfig,
    RunStudyStores,
    run_study,
)
from vip.application.screen_factors import ScreenConfig
from vip.domain.errors import PersistenceError
from vip.domain.value_objects import DateRange, Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore

_N_OHLCV_ROWS = 240
_N_SPLITS = 2
_N_REPEATS = 1
_TOP_K = 1
_EMBARGO = 5
_RANDOM_SEED = 0
_HORIZON_DEFAULT = 5
_HORIZON_STALE = 21


def _synthetic_ohlcv(n_rows: int = _N_OHLCV_ROWS) -> pd.DataFrame:
    """Build synthetic canonical OHLCV."""
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


def _stale_matrix_wrong_horizon() -> pd.DataFrame:
    """Minimal matrix with only ``target_rv_cc_21d`` (post screen-horizons)."""
    index = pd.bdate_range("2024-01-02", periods=_N_OHLCV_ROWS)
    frame = pd.DataFrame(
        {
            "rv_cc_1d": np.linspace(0.01, 0.05, _N_OHLCV_ROWS),
            "rv_cc_5d": np.linspace(0.02, 0.06, _N_OHLCV_ROWS),
            "rv_cc_21d": np.linspace(0.03, 0.07, _N_OHLCV_ROWS),
            "ret_1d": np.zeros(_N_OHLCV_ROWS),
            "ret_5d": np.zeros(_N_OHLCV_ROWS),
            "range_1d": np.linspace(0.005, 0.02, _N_OHLCV_ROWS),
            "range_5d_mean": np.linspace(0.005, 0.02, _N_OHLCV_ROWS),
            "volume_z_21d": np.zeros(_N_OHLCV_ROWS),
            f"target_rv_cc_{_HORIZON_STALE}d": np.linspace(
                0.04, 0.08, _N_OHLCV_ROWS,
            ),
        },
        index=index,
    )
    return frame


@dataclass(frozen=True, slots=True)
class _NoOpMarketDataSource:
    """MarketDataSource stub that must not be called under skip-ingest."""

    def fetch(self, symbol: Symbol, date_range: DateRange) -> pd.DataFrame:
        """Refuse unexpected ingest."""
        raise AssertionError("Ingestion should be skipped in this test.")

    def source_name(self) -> str:
        """Return a stable source name."""
        return "noop"


def _build_stores(tmp_path: Path) -> RunStudyStores:
    """Wire parquet / artifact stores under ``tmp_path``."""
    return RunStudyStores(
        source=_NoOpMarketDataSource(),
        market_store=ParquetMarketDataStore(tmp_path / "raw"),
        feature_store=ParquetFeatureMatrixStore(tmp_path / "processed"),
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
    )


def _screen_config() -> ScreenConfig:
    """Tiny walk-forward settings for offline tests."""
    return ScreenConfig(
        n_splits=_N_SPLITS,
        embargo_size=_EMBARGO,
        n_repeats=_N_REPEATS,
        top_k=_TOP_K,
        random_seed=_RANDOM_SEED,
    )


def test_run_rebuilds_stale_wrong_horizon_target(tmp_path: Path) -> None:
    """Existing 21d-only matrix is rebuilt to 5d when skip_features is False."""
    stores = _build_stores(tmp_path)
    symbol = Symbol("SPY")
    stores.market_store.save(symbol, _synthetic_ohlcv())
    stores.feature_store.save(symbol, _stale_matrix_wrong_horizon())
    assert f"target_rv_cc_{_HORIZON_STALE}d" in stores.feature_store.load(
        symbol,
    ).columns

    config = RunStudyConfig(
        symbols=[symbol],
        date_range=DateRange(start=date(2024, 1, 2), end=date(2024, 4, 1)),
        horizon_days=_HORIZON_DEFAULT,
        skip_ingest=True,
        skip_features=False,
        screen_config=_screen_config(),
    )
    run_study(stores=stores, config=config)

    loaded = stores.feature_store.load(symbol)
    assert f"target_rv_cc_{_HORIZON_DEFAULT}d" in loaded.columns


def test_run_skip_features_wrong_target_raises(tmp_path: Path) -> None:
    """skip_features with wrong target raises a clear PersistenceError."""
    stores = _build_stores(tmp_path)
    symbol = Symbol("SPY")
    stores.market_store.save(symbol, _synthetic_ohlcv())
    stores.feature_store.save(symbol, _stale_matrix_wrong_horizon())

    config = RunStudyConfig(
        symbols=[symbol],
        date_range=DateRange(start=date(2024, 1, 2), end=date(2024, 4, 1)),
        horizon_days=_HORIZON_DEFAULT,
        skip_ingest=True,
        skip_features=True,
        screen_config=_screen_config(),
    )
    with pytest.raises(PersistenceError, match="target_rv_cc_5d") as exc_info:
        run_study(stores=stores, config=config)
    message = str(exc_info.value)
    assert "vip features" in message
    assert "--horizon 5" in message


def test_run_rebuild_with_iv_rv_adds_gap_columns(tmp_path: Path) -> None:
    """Rebuilding with include_iv_rv persists gap columns used by screen."""
    stores = _build_stores(tmp_path)
    symbol = Symbol("SPY")
    stores.market_store.save(symbol, _synthetic_ohlcv())
    stores.market_store.save(Symbol("VIX"), _synthetic_ohlcv())
    stores.feature_store.save(symbol, _stale_matrix_wrong_horizon())

    config = RunStudyConfig(
        symbols=[symbol],
        date_range=DateRange(start=date(2024, 1, 2), end=date(2024, 4, 1)),
        horizon_days=_HORIZON_DEFAULT,
        extras=FeatureMatrixExtras(include_iv_rv=True),
        skip_ingest=True,
        skip_features=False,
        screen_config=_screen_config(),
    )
    run_study(stores=stores, config=config)

    loaded = stores.feature_store.load(symbol)
    assert f"target_rv_cc_{_HORIZON_DEFAULT}d" in loaded.columns
    assert {
        "vix_minus_rv_5d",
        "vix_rv_ratio_5d",
        "vix_vol_daily",
    } <= set(loaded.columns)