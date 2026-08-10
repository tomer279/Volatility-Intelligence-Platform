"""Horse-race wiring for Milestone 9 ``vix_as_forecast``.

Tests
-----
test_horse_race_models_include_vix_as_forecast
    Catalog constant lists the locked IV forecast name.
test_screen_factors_runs_vix_as_forecast_when_vix_present
    Synthetic panel with ``vix_level`` yields summary + inference rows.
test_screen_factors_skips_vix_as_forecast_without_vix_columns
    Panels without VIX predictors keep the legacy three-model race.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vip.application.screen_horse_race import (
    HORSE_RACE_MODELS,
    VIX_AS_FORECAST_MODEL
)
from vip.application.screen_factors import (
    ScreenConfig,
    ScreenInferenceOptions,
    screen_factors
)
from vip.domain.value_objects import Symbol
from vip.evaluation.inference import BootstrapInferenceOptions
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

N_ROWS = 240


def _base_features(rng: np.random.Generator, index: pd.DatetimeIndex) -> pd.DataFrame:
    """HAR-style covariates shared by wiring fixtures."""
    return pd.DataFrame(
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


def _with_target(features: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Attach synthetic ``target_rv_cc_5d``."""
    frame = features.copy()
    frame["target_rv_cc_5d"] = (
        0.05
        + 0.5 * frame["rv_cc_1d"]
        + 0.3 * frame["rv_cc_5d"]
        + 0.2 * frame["rv_cc_21d"]
        + rng.normal(0.0, 0.001, N_ROWS)
    )
    return frame


def _matrix_without_vix() -> pd.DataFrame:
    """Feature matrix without VIX predictors."""
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    rng = np.random.default_rng(7)
    return _with_target(_base_features(rng, index), rng)


def _matrix_with_vix_level() -> pd.DataFrame:
    """Feature matrix including ``vix_level`` for ``vix_as_forecast``."""
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    rng = np.random.default_rng(11)
    features = _base_features(rng, index)
    features["vix_level"] = rng.uniform(12.0, 35.0, N_ROWS)
    return _with_target(features, rng)


def _cheap_inference() -> ScreenInferenceOptions:
    """Small bootstrap for fast unit tests."""
    return ScreenInferenceOptions(
        bootstrap=BootstrapInferenceOptions(
            block_length=10,
            n_resamples=99,
            alpha=0.05,
            random_seed=0,
        ),
        include_hln_dm=False,
        include_nonoverlap_sensitivity=False,
    )


def _cheap_config() -> ScreenConfig:
    """Walk-forward settings matching other screen unit tests."""
    return ScreenConfig(
        n_splits=3,
        embargo_size=5,
        n_repeats=2,
        top_k=3,
        random_seed=0,
    )


def test_horse_race_models_include_vix_as_forecast() -> None:
    """Locked catalog must list ``vix_as_forecast``."""
    assert VIX_AS_FORECAST_MODEL in HORSE_RACE_MODELS


def test_screen_factors_runs_vix_as_forecast_when_vix_present(
        tmp_path: Path,
) -> None:
    """With ``vix_level``, summary and inference.json include the IV model."""
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    symbol = Symbol("SPY")
    feature_store.save(symbol, _matrix_with_vix_level())

    result = screen_factors(
        feature_store=feature_store,
        artifact_store=artifact_store,
        symbol=symbol,
        config=_cheap_config(),
        inference=_cheap_inference(),
    )

    models = set(result.tables.summary["model"].astype(str))
    assert VIX_AS_FORECAST_MODEL in models
    assert "har_rv_ols" in models

    experiment_dir = (
        tmp_path / "artifacts" / result.identity.experiment_id.as_path_key()
    )
    inference_path = experiment_dir / "inference.json"
    assert inference_path.is_file()
    inference_rows = json.loads(inference_path.read_text(encoding="utf-8"))
    inference_models = {str(row["model"]) for row in inference_rows}
    assert VIX_AS_FORECAST_MODEL in inference_models

    vix_row = result.tables.summary.loc[
        result.tables.summary["model"] == VIX_AS_FORECAST_MODEL
    ].iloc[0]
    assert "mean_delta_qlike" in result.tables.summary.columns
    assert pd.notna(vix_row["mean_delta_qlike"])
    assert "bootstrap_pvalue" in result.tables.summary.columns


def test_screen_factors_skips_vix_as_forecast_without_vix_columns(
        tmp_path: Path,
) -> None:
    """Legacy panels without VIX keep the three-model race."""
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    symbol = Symbol("SPY")
    feature_store.save(symbol, _matrix_without_vix())

    result = screen_factors(
        feature_store=feature_store,
        artifact_store=artifact_store,
        symbol=symbol,
        config=_cheap_config(),
        inference=_cheap_inference(),
    )

    assert set(result.tables.summary["model"].astype(str)) == {
        "har_rv_ols",
        "ridge",
        "lasso",
    }