"""Horizon-injectable factor-screen smoke tests (network-free)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vip.application.screen_factors import (
    ScreenConfig,
    ScreenInferenceOptions,
    screen_factors,
    settings_for_horizon,
    target_column_for_horizon,
)
from vip.domain.value_objects import Symbol
from vip.evaluation.inference import BootstrapInferenceOptions
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

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


def _fast_screen_settings(
        horizon_days: int
    ) -> tuple[ScreenConfig, ScreenInferenceOptions]:
    """Cheap walk-forward / bootstrap settings for unit tests."""
    defaults = settings_for_horizon(horizon_days)
    config = replace(
        defaults.config,
        n_splits=_N_SPLITS,
        n_repeats=_N_REPEATS,
        top_k=_TOP_K,
        random_seed=0,
    )
    inference = replace(
        defaults.inference,
        include_hln_dm=False,
        include_nonoverlap_sensitivity=False,
        bootstrap=replace(
            defaults.inference.bootstrap,
            n_resamples=_BOOTSTRAP_N_RESAMPLES,
            random_seed=0,
        ),
    )
    return config, inference


@pytest.mark.parametrize("horizon_days", [1, 21])
def test_screen_factors_injects_horizon_and_embargo(
    tmp_path: Path,
    horizon_days: int,
) -> None:
    """Screen uses ``target_rv_cc_{h}d`` and injectable ``embargo_size = h``."""
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    symbol = Symbol("SPY")
    feature_store.save(symbol, _synthetic_matrix(horizon_days))

    config, inference = _fast_screen_settings(horizon_days)
    assert config.embargo_size == horizon_days

    result = screen_factors(
        feature_store=feature_store,
        artifact_store=artifact_store,
        symbol=symbol,
        config=config,
        inference=inference,
    )

    meta = artifact_store.read_json(result.identity.experiment_id, "screen_meta")
    expected_target = target_column_for_horizon(horizon_days)
    assert meta["horizon_days"] == horizon_days
    assert meta["target_column"] == expected_target
    assert meta["embargo_size"] == horizon_days
    assert meta["nw_lags"] == horizon_days - 1
    assert not result.tables.summary.empty


def test_target_column_for_horizon_names() -> None:
    """Target naming matches the locked ``target_rv_cc_{h}d`` contract."""
    assert target_column_for_horizon(1) == "target_rv_cc_1d"
    assert target_column_for_horizon(5) == "target_rv_cc_5d"
    assert target_column_for_horizon(21) == "target_rv_cc_21d"


def test_default_screen_still_uses_five_day_target(tmp_path: Path) -> None:
    """h=5 inference settings keep target ``target_rv_cc_5d`` and ``horizon_days`` 5."""
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    symbol = Symbol("SPY")
    feature_store.save(symbol, _synthetic_matrix(5))

    result = screen_factors(
        feature_store=feature_store,
        artifact_store=artifact_store,
        symbol=symbol,
        config=ScreenConfig(
            n_splits=_N_SPLITS,
            embargo_size=5,
            n_repeats=_N_REPEATS,
            top_k=_TOP_K,
            random_seed=0,
        ),
        inference=replace(
            settings_for_horizon(5).inference,
            include_hln_dm=False,
            include_nonoverlap_sensitivity=False,
            bootstrap=BootstrapInferenceOptions(
                block_length=10,
                n_resamples=_BOOTSTRAP_N_RESAMPLES,
                alpha=0.05,
                random_seed=0,
            ),
        ),
    )
    meta = artifact_store.read_json(result.identity.experiment_id, "screen_meta")
    assert meta["target_column"] == "target_rv_cc_5d"
    assert meta["horizon_days"] == 5
