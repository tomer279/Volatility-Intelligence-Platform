"""Tests for feature-matrix application use-case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vip.application.build_feature_matrix import (
    build_and_persist_feature_matrix,
    FeatureMatrixExtras,
)
from vip.cli.feature_extras import parse_feature_extras
from vip.domain.value_objects import Symbol
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore


def _synthetic_ohlcv(n_rows: int = 80) -> pd.DataFrame:
    """Build synthetic canonical OHLCV data."""
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


def test_build_and_persist_feature_matrix(tmp_path: Path) -> None:
    """Use-case should load OHLCV, build matrix, and persist features."""
    market_root = tmp_path / "raw"
    feature_root = tmp_path / "processed"
    market_store = ParquetMarketDataStore(market_root)
    feature_store = ParquetFeatureMatrixStore(feature_root)

    symbol = Symbol("SPY")
    market_store.save(symbol, _synthetic_ohlcv())

    result = build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=symbol,
        horizon_days=5,
    )

    assert result.symbol == symbol
    assert result.row_count > 0
    assert result.feature_count == 8
    assert result.output_path.is_file()
    assert feature_store.exists(symbol)

    loaded = feature_store.load(symbol)
    assert "target_rv_cc_5d" in loaded.columns
    assert not loaded.isna().any().any()


def test_build_with_vix_increases_feature_count(tmp_path: Path) -> None:
    market_store = ParquetMarketDataStore(tmp_path / "raw")
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    market_store.save(Symbol("SPY"), _synthetic_ohlcv())
    market_store.save(Symbol("VIX"), _synthetic_ohlcv())  # synthetic stand-in

    result = build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=Symbol("SPY"),
        horizon_days=5,
        extras=FeatureMatrixExtras(include_vix=True),
    )
    assert result.feature_count == 10  # 8 own-symbol + 2 VIX
    loaded = feature_store.load(Symbol("SPY"))
    assert {"vix_level", "vix_chg_1d"} <= set(loaded.columns)
    assert "vix_vol_daily" not in loaded.columns
    assert "vix_minus_rv_5d" not in loaded.columns


def test_build_with_jump_increases_feature_count(tmp_path: Path) -> None:
    """Jump family adds three proportion columns (not bipower levels)."""
    market_store = ParquetMarketDataStore(tmp_path / "raw")
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    market_store.save(Symbol("SPY"), _synthetic_ohlcv())

    result = build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=Symbol("SPY"),
        horizon_days=5,
        extras=FeatureMatrixExtras(include_jump=True),
    )
    assert result.feature_count == 11  # 8 core + 3 jump_prop
    loaded = feature_store.load(Symbol("SPY"))
    assert {"jump_prop_1d", "jump_prop_5d", "jump_prop_21d"} <= set(loaded.columns)
    assert not any(c.startswith("bpv_cc_") for c in loaded.columns)


def test_parse_feature_extras_iv_rv_implies_vix() -> None:
    """CLI token iv_rv sets include_iv_rv and implies include_vix."""
    extras = parse_feature_extras("iv_rv")
    assert extras.include_iv_rv is True
    assert extras.include_vix is True
    assert extras.include_jump is False


def test_parse_feature_extras_vix_alone_no_iv_rv() -> None:
    """Bare vix must not enable the IV−RV family (ablation)."""
    extras = parse_feature_extras("vix")
    assert extras.include_vix is True
    assert extras.include_iv_rv is False


def test_build_with_iv_rv_adds_gap_columns(tmp_path: Path) -> None:
    """IV−RV path joins VIX and appends gap columns when VIX OHLCV exists."""
    market_store = ParquetMarketDataStore(tmp_path / "raw")
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    market_store.save(Symbol("SPY"), _synthetic_ohlcv())
    market_store.save(Symbol("VIX"), _synthetic_ohlcv())

    result = build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=Symbol("SPY"),
        horizon_days=5,
        extras=FeatureMatrixExtras(include_iv_rv=True),
    )
    # 8 core + 2 VIX + 5 iv_rv
    assert result.feature_count == 15
    loaded = feature_store.load(Symbol("SPY"))
    assert {
        "vix_level",
        "vix_chg_1d",
        "vix_vol_daily",
        "vix_minus_rv_1d",
        "vix_minus_rv_5d",
        "vix_minus_rv_21d",
        "vix_rv_ratio_5d",
    } <= set(loaded.columns)


def test_build_with_vix_alone_excludes_gap_columns(tmp_path: Path) -> None:
    """Ablation: include_vix must not add IV−RV gap columns."""
    market_store = ParquetMarketDataStore(tmp_path / "raw")
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    market_store.save(Symbol("SPY"), _synthetic_ohlcv())
    market_store.save(Symbol("VIX"), _synthetic_ohlcv())

    build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=Symbol("SPY"),
        horizon_days=5,
        extras=FeatureMatrixExtras(include_vix=True),
    )
    loaded = feature_store.load(Symbol("SPY"))
    assert {"vix_level", "vix_chg_1d"} <= set(loaded.columns)
    assert "vix_vol_daily" not in loaded.columns
    assert "vix_minus_rv_1d" not in loaded.columns
    assert "vix_minus_rv_5d" not in loaded.columns
    assert "vix_minus_rv_21d" not in loaded.columns


def test_parse_feature_extras_rates() -> None:
    """CLI token rates sets include_rates without implying VIX."""
    extras = parse_feature_extras("rates")
    assert extras.include_rates is True
    assert extras.include_vix is False
    assert extras.include_iv_rv is False


def test_build_with_rates_adds_tnx_columns(tmp_path: Path) -> None:
    """Rates path joins TNX when TNX OHLCV exists."""
    market_store = ParquetMarketDataStore(tmp_path / "raw")
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    market_store.save(Symbol("SPY"), _synthetic_ohlcv())
    market_store.save(Symbol("TNX"), _synthetic_ohlcv())

    result = build_and_persist_feature_matrix(
        market_store=market_store,
        feature_store=feature_store,
        symbol=Symbol("SPY"),
        horizon_days=5,
        extras=FeatureMatrixExtras(include_rates=True),
    )
    # 8 core + 2 rates
    assert result.feature_count == 10
    loaded = feature_store.load(Symbol("SPY"))
    assert {"tnx_level", "tnx_chg_1d"} <= set(loaded.columns)