"""Tests for the factor-screening application use-case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vip.application.screen_factors import ScreenConfig, screen_factors
from vip.domain.value_objects import Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

N_ROWS = 240


def _synthetic_matrix() -> pd.DataFrame:
    """Build a synthetic processed feature matrix."""
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


def test_screen_factors_persists_artifacts(tmp_path: Path) -> None:
    """Screening should write metrics/importance/ranking artifacts."""
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    symbol = Symbol("SPY")
    feature_store.save(symbol, _synthetic_matrix())

    result = screen_factors(
        feature_store=feature_store,
        artifact_store=artifact_store,
        symbol=symbol,
        config=ScreenConfig(
            n_splits=3,
            embargo_size=5,
            n_repeats=2,
            top_k=3,
            random_seed=0,
        ),
    )

    experiment_dir = (
        tmp_path / "artifacts" / result.identity.experiment_id.as_path_key()
    )
    assert result.identity.screening_model == "ridge"
    assert result.top_feature() in set(result.tables.ranking["feature"])
    assert not result.tables.summary.empty
    assert set(result.tables.summary["model"]) == {"har_rv_ols", "ridge", "lasso"}
    for name in (
        "metrics.json",
        "folds.json",
        "oos_losses.json",
        "inference.json",
        "inference_sensitivity.json",
        "importance.json",
        "factor_ranking.json",
        "metrics_by_regime.json",
        "screen_meta.json",
        "importance_plot.png",
        "report.html",
    ):
        assert (experiment_dir / name).is_file()

    assert "mean_delta_qlike" in result.tables.summary.columns
    assert "bootstrap_pvalue" in result.tables.summary.columns