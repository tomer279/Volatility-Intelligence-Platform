"""Tests for baseline experiment use-case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vip.application.run_baseline_experiment import run_baseline_experiment
from vip.domain.value_objects import Symbol
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore


def _synthetic_matrix(n_rows: int = 240) -> pd.DataFrame:
    """Build a synthetic processed feature matrix."""
    index = pd.bdate_range("2020-01-01", periods=n_rows)
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
        {
            "rv_cc_1d": rng.uniform(0.01, 0.05, n_rows),
            "rv_cc_5d": rng.uniform(0.02, 0.06, n_rows),
            "rv_cc_21d": rng.uniform(0.03, 0.07, n_rows),
            "ret_1d": rng.normal(0.0, 0.01, n_rows),
            "ret_5d": rng.normal(0.0, 0.02, n_rows),
            "range_1d": rng.uniform(0.005, 0.02, n_rows),
            "range_5d_mean": rng.uniform(0.005, 0.02, n_rows),
            "volume_z_21d": rng.normal(0.0, 1.0, n_rows),
        },
        index=index,
    )
    target = (
        0.05
        + 0.5 * features["rv_cc_1d"]
        + 0.3 * features["rv_cc_5d"]
        + 0.2 * features["rv_cc_21d"]
    )
    features["target_rv_cc_5d"] = target
    return features


def test_run_baseline_experiment_persists_artifacts(tmp_path: Path) -> None:
    """Use-case should evaluate baselines and write metrics artifacts."""
    feature_store = ParquetFeatureMatrixStore(tmp_path / "processed")
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    symbol = Symbol("SPY")
    feature_store.save(symbol, _synthetic_matrix())

    result = run_baseline_experiment(
        feature_store=feature_store,
        artifact_store=artifact_store,
        symbol=symbol,
        n_splits=4,
        embargo_size=5,
    )

    assert result.primary_model in {"historical_mean", "ewma", "har_rv_ols"}
    assert not result.summary.empty
    assert (tmp_path / "artifacts" / result.experiment_id.as_path_key() / "metrics.json").is_file()
    assert (tmp_path / "artifacts" / result.experiment_id.as_path_key() / "folds.json").is_file()