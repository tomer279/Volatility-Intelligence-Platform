"""Horse-race wiring for Milestone 10 ``ou_rv``.

Tests
-----
test_horse_race_models_include_ou_rv
    Catalog constant lists the locked OU forecast name.
test_default_registry_includes_ou_rv
    Default factory is zero-arg and uses horizon_days=5.
test_screen_factors_runs_ou_rv_in_summary_and_inference
    Synthetic panel yields summary + inference.json rows for ``ou_rv``.
test_vix_absent_matrix_still_includes_ou_rv
    Panels without VIX keep ``ou_rv`` and drop only ``vix_as_forecast``.
test_resolve_injects_non_default_horizon_into_ou_rv
    ``resolve_horse_race_models`` passes h ∈ {1, 21} into ``OuRvModel``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vip.application.screen_horse_race import (
    HORSE_RACE_MODELS,
    OU_RV_MODEL,
    VIX_AS_FORECAST_MODEL,
    resolve_horse_race_models,
)
from vip.application.screen_factors import (
    ScreenConfig,
    ScreenInferenceOptions,
    screen_factors,
)
from vip.domain.value_objects import Symbol
from vip.evaluation.inference import BootstrapInferenceOptions
from vip.modeling.baselines import DEFAULT_OU_HORIZON_DAYS, OuRvModel
from vip.modeling.registry import create_default_model_registry
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

N_ROWS = 240
HORIZON_ONE = 1
HORIZON_TWENTY_ONE = 21
TINY_N_ROWS = 4


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
    """Feature matrix including ``vix_level``."""
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    rng = np.random.default_rng(11)
    features = _base_features(rng, index)
    features["vix_level"] = rng.uniform(12.0, 35.0, N_ROWS)
    return _with_target(features, rng)


def _tiny_features_without_vix() -> pd.DataFrame:
    """Short frame for resolve-only tests (no fit)."""
    index = pd.bdate_range("2020-01-01", periods=TINY_N_ROWS)
    return pd.DataFrame(
        {"rv_cc_1d": [0.01, 0.02, 0.015, 0.018]},
        index=index,
    )


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


def test_horse_race_models_include_ou_rv() -> None:
    """Locked catalog must list ``ou_rv`` unconditionally."""
    assert OU_RV_MODEL in HORSE_RACE_MODELS


def test_default_registry_includes_ou_rv() -> None:
    """Default factory is zero-arg and uses the locked h=5 default."""
    registry = create_default_model_registry()
    assert OU_RV_MODEL in registry.list_names()
    model = registry.create(OU_RV_MODEL)
    assert isinstance(model, OuRvModel)
    assert model.horizon_days() == DEFAULT_OU_HORIZON_DAYS


def test_screen_factors_runs_ou_rv_in_summary_and_inference(
        tmp_path: Path,
) -> None:
    """Summary and inference.json include ``ou_rv`` (VIX present)."""
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
    assert OU_RV_MODEL in models
    assert "har_rv_ols" in models
    assert VIX_AS_FORECAST_MODEL in models

    experiment_dir = (
        tmp_path / "artifacts" / result.identity.experiment_id.as_path_key()
    )
    inference_path = experiment_dir / "inference.json"
    assert inference_path.is_file()
    inference_rows = json.loads(inference_path.read_text(encoding="utf-8"))
    inference_models = {str(row["model"]) for row in inference_rows}
    assert OU_RV_MODEL in inference_models

    ou_row = result.tables.summary.loc[
        result.tables.summary["model"] == OU_RV_MODEL
    ].iloc[0]
    assert "mean_delta_qlike" in result.tables.summary.columns
    assert pd.notna(ou_row["mean_delta_qlike"])
    assert "bootstrap_pvalue" in result.tables.summary.columns


def test_vix_absent_matrix_still_includes_ou_rv(tmp_path: Path) -> None:
    """Without VIX columns, drop only ``vix_as_forecast``; keep ``ou_rv``."""
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

    models = set(result.tables.summary["model"].astype(str))
    assert models == {
        "har_rv_ols", "ridge", "lasso", OU_RV_MODEL, "ewma_recursive",
    }
    assert VIX_AS_FORECAST_MODEL not in models

    resolve_models = resolve_horse_race_models(_tiny_features_without_vix())
    assert OU_RV_MODEL in resolve_models
    assert VIX_AS_FORECAST_MODEL not in resolve_models


def test_resolve_injects_non_default_horizon_into_ou_rv() -> None:
    """M8 horizons 1 and 21 must reach ``OuRvModel``, not a hard-coded 5."""
    features = _tiny_features_without_vix()
    models_h1 = resolve_horse_race_models(features, horizon_days=HORIZON_ONE)
    models_h21 = resolve_horse_race_models(
        features,
        horizon_days=HORIZON_TWENTY_ONE,
    )
    ou_h1 = models_h1[OU_RV_MODEL]
    ou_h21 = models_h21[OU_RV_MODEL]
    assert isinstance(ou_h1, OuRvModel)
    assert isinstance(ou_h21, OuRvModel)
    assert ou_h1.horizon_days() == HORIZON_ONE
    assert ou_h21.horizon_days() == HORIZON_TWENTY_ONE
    assert VIX_AS_FORECAST_MODEL not in models_h1