"""Horse-race wiring for Milestone 10 stretch ``ewma_recursive``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vip.application.screen_horse_race import (
    EWMA_RECURSIVE_MODEL,
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
from vip.modeling.parametric import EwmaRecursiveModel
from vip.modeling.registry import create_default_model_registry
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

N_ROWS = 240


def _matrix_without_vix() -> pd.DataFrame:
    """Synthetic panel without VIX predictors."""
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
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
    frame["target_rv_cc_5d"] = (
        0.05
        + 0.5 * frame["rv_cc_1d"]
        + 0.3 * frame["rv_cc_5d"]
        + 0.2 * frame["rv_cc_21d"]
        + rng.normal(0.0, 0.001, N_ROWS)
    )
    return frame


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


def test_horse_race_models_include_ewma_recursive() -> None:
    """Locked catalog must list ``ewma_recursive`` unconditionally."""
    assert EWMA_RECURSIVE_MODEL in HORSE_RACE_MODELS


def test_default_registry_includes_ewma_recursive() -> None:
    """Default factory is zero-arg EwmaRecursiveModel."""
    registry = create_default_model_registry()
    assert EWMA_RECURSIVE_MODEL in registry.list_names()
    model = registry.create(EWMA_RECURSIVE_MODEL)
    assert isinstance(model, EwmaRecursiveModel)


def test_screen_includes_ewma_recursive_in_summary_and_inference(
        tmp_path: Path,
) -> None:
    """Summary and inference.json include ``ewma_recursive``."""
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
    assert EWMA_RECURSIVE_MODEL in models
    assert OU_RV_MODEL in models
    assert VIX_AS_FORECAST_MODEL not in models

    experiment_dir = (
        tmp_path / "artifacts" / result.identity.experiment_id.as_path_key()
    )
    inference_rows = json.loads(
        (experiment_dir / "inference.json").read_text(encoding="utf-8")
    )
    inference_models = {str(row["model"]) for row in inference_rows}
    assert EWMA_RECURSIVE_MODEL in inference_models

    resolve_models = resolve_horse_race_models(_matrix_without_vix())
    assert EWMA_RECURSIVE_MODEL in resolve_models