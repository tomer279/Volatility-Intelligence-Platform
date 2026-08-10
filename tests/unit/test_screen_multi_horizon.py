"""Multi-horizon screen orchestration tests (network-free)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import json
import numpy as np
import pandas as pd
import pytest

from vip.application.build_feature_matrix import (
    BuildFeatureMatrixResult,
    FeatureMatrixExtras
)
from vip.application.screen_factors import ScreenConfig, target_column_for_horizon
from vip.application.screen_multi_horizon import (
    HORIZON_SUMMARY_COLUMNS,
    MultiHorizonInferenceOverrides,
    MultiHorizonScreenConfig,
    MultiHorizonStores,
    screen_multi_horizon,
)
from vip.domain.value_objects import Symbol
from vip.evaluation.horizon_defaults import LOCKED_SCREEN_HORIZONS
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore
from vip.domain.errors import PersistenceError


N_ROWS = 240
_N_SPLITS = 3
_N_REPEATS = 2
_TOP_K = 3
_BOOTSTRAP_N_RESAMPLES = 99


def _synthetic_matrix(horizon_days: int) -> pd.DataFrame:
    """Build a synthetic matrix with one forward-RV target column."""
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
    features[target_column_for_horizon(horizon_days)] = target
    return features


def _fake_build(**kwargs):
    """Write a synthetic one-target matrix (no OHLCV / network).

    Accepts the same keyword arguments as
    ``build_and_persist_feature_matrix``; only ``feature_store``,
    ``symbol``, and ``horizon_days`` are used.
    """
    feature_store = kwargs["feature_store"]
    symbol = kwargs["symbol"]
    horizon_days = int(kwargs.get("horizon_days", 5))
    frame = _synthetic_matrix(horizon_days)
    path = feature_store.save(symbol, frame)
    return BuildFeatureMatrixResult(
        symbol=symbol,
        row_count=int(frame.shape[0]),
        feature_count=int(frame.shape[1] - 1),
        output_path=path,
        start_date=frame.index.min().date().isoformat(),
        end_date=frame.index.max().date().isoformat(),
    )


def _build_stores(tmp_path: Path) -> MultiHorizonStores:
    """Build temporary market/feature/artifact stores under ``tmp_path``."""
    return MultiHorizonStores(
        market_store=ParquetMarketDataStore(tmp_path / "raw"),
        feature_store=ParquetFeatureMatrixStore(tmp_path / "processed"),
        artifact_root=tmp_path / "artifacts",
    )


def test_screen_multi_horizon_writes_summary_and_layout(
        tmp_path: Path,
) -> None:
    """Study root gets meta/summary/HTML; each h{h}d has metrics.json."""
    stores = _build_stores(tmp_path)
    symbol = Symbol("SPY")
    config = MultiHorizonScreenConfig(
        symbol=symbol,
        feature_extras=FeatureMatrixExtras(include_vix = True),
        horizons=LOCKED_SCREEN_HORIZONS,
        skip_features=False,
        screen_config=ScreenConfig(
            n_splits=_N_SPLITS,
            n_repeats=_N_REPEATS,
            top_k=_TOP_K,
            random_seed=0,
        ),
        inference=MultiHorizonInferenceOverrides(
            bootstrap_n_resamples=_BOOTSTRAP_N_RESAMPLES,
            include_hln_dm=False,
            include_nonoverlap_sensitivity=False,
        ),
    )
    with patch(
        "vip.application.screen_multi_horizon.build_and_persist_feature_matrix",
        side_effect=_fake_build,
    ):
        result = screen_multi_horizon(
            stores=stores,
            config=config,
        )

    study_dir = stores.resolve_study_dir(result.study_id)
    assert (study_dir / "horizon_summary.json").is_file()
    assert (study_dir / "screen_meta.json").is_file()
    assert (study_dir / "report.html").is_file()

    html = (study_dir / "report.html").read_text(encoding="utf-8")
    assert "Skill by horizon" in html
    for column in HORIZON_SUMMARY_COLUMNS:
        assert column in result.summary.columns

    for horizon_days in LOCKED_SCREEN_HORIZONS:
        horizon_dir = study_dir / f"h{horizon_days}d"
        assert (horizon_dir / "metrics.json").is_file()
        assert (horizon_dir / "screen_meta.json").is_file()
        # nested factor-screen-* folder should be promoted away
        nested = list(horizon_dir.glob("factor-screen-*"))
        assert not nested

    assert set(result.summary["horizon_days"].unique()) == set(LOCKED_SCREEN_HORIZONS)
    # screen_meta is a dict — load via json for robustness in your apply step
    meta_obj = json.loads((study_dir / "screen_meta.json").read_text(encoding="utf-8"))
    assert meta_obj["horizons"] == list(LOCKED_SCREEN_HORIZONS)
    assert len(meta_obj["per_horizon"]) == len(LOCKED_SCREEN_HORIZONS)
    for block in meta_obj["per_horizon"]:
        h = int(block["horizon_days"])
        assert block["embargo_size"] == h
        assert block["nw_lags"] == h - 1


def test_skip_features_requires_matching_target(
        tmp_path: Path,
    ) -> None:
    """skip_features fails when the cached target column is wrong."""
    stores = _build_stores(tmp_path)
    symbol = Symbol("SPY")
    stores.feature_store.save(symbol, _synthetic_matrix(5))
    config = MultiHorizonScreenConfig(
        symbol=symbol,
        horizons=(1,),
        skip_features=True,
        screen_config=ScreenConfig(n_splits=_N_SPLITS, n_repeats=_N_REPEATS),
        inference=MultiHorizonInferenceOverrides(
            bootstrap_n_resamples=_BOOTSTRAP_N_RESAMPLES,
            include_hln_dm=False,
            include_nonoverlap_sensitivity=False,
        ),
    )
    with pytest.raises(PersistenceError):
        screen_multi_horizon(stores=stores, config=config)
