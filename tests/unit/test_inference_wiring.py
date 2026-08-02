"""Wiring tests for inference-enriched comparison / screen artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vip.application.screen_factors import (
    ScreenConfig,
    ScreenInferenceOptions,
    screen_factors,
)
from vip.domain.value_objects import Symbol
from vip.evaluation.comparison import (
    InferenceSummaryOptions,
    summarize_with_inference,
)
from vip.evaluation.inference import BootstrapInferenceOptions
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

N_DATES = 40
N_SCREEN_ROWS = 240


def _tiny_fold_metrics() -> pd.DataFrame:
    """Two-fold descriptive metrics for baseline + one challenger."""
    return pd.DataFrame(
        [
            {"model": "har_rv_ols", "fold_id": 0, "qlike": 1.0, "mse": 0.2, "mae": 0.1},
            {"model": "har_rv_ols", "fold_id": 1, "qlike": 1.2, "mse": 0.3, "mae": 0.15},
            {"model": "ridge", "fold_id": 0, "qlike": 0.8, "mse": 0.18, "mae": 0.09},
            {"model": "ridge", "fold_id": 1, "qlike": 0.9, "mse": 0.22, "mae": 0.11},
        ]
    )


def _tiny_oos_losses() -> pd.DataFrame:
    """Aligned OOS loss panel with a clear negative ridge gap."""
    index = pd.bdate_range("2020-01-01", periods=N_DATES)
    rng = np.random.default_rng(0)
    baseline = 1.0 + rng.normal(0.0, 0.05, N_DATES)
    challenger = 0.7 + rng.normal(0.0, 0.05, N_DATES)
    frames = [
        pd.DataFrame(
            {
                "model": "har_rv_ols",
                "fold_id": 0,
                "y_true": 0.05,
                "y_pred": 0.05,
                "qlike_loss": baseline,
            },
            index=index,
        ),
        pd.DataFrame(
            {
                "model": "ridge",
                "fold_id": 0,
                "y_true": 0.05,
                "y_pred": 0.04,
                "qlike_loss": challenger,
            },
            index=index,
        ),
    ]
    return pd.concat(frames, axis=0)


def test_summarize_with_inference_fills_challenger_only() -> None:
    """Baseline row blanks; challenger gets mean Δ / CI / p."""
    options = InferenceSummaryOptions(
        bootstrap=BootstrapInferenceOptions(
            block_length=10,
            n_resamples=199,
            alpha=0.05,
            random_seed=0,
        ),
        include_hln_dm=True,
        horizon_days=5,
    )
    summary = summarize_with_inference(
        _tiny_fold_metrics(),
        _tiny_oos_losses(),
        options=options,
    )
    assert list(summary["model"]) == ["ridge", "har_rv_ols"] or set(
        summary["model"]
    ) == {"ridge", "har_rv_ols"}

    baseline = summary.loc[summary["model"] == "har_rv_ols"].iloc[0]
    challenger = summary.loc[summary["model"] == "ridge"].iloc[0]

    assert baseline["mean_delta_qlike"] is None or pd.isna(baseline["mean_delta_qlike"])
    assert baseline["bootstrap_pvalue"] is None or pd.isna(baseline["bootstrap_pvalue"])
    assert challenger["mean_delta_qlike"] == pytest.approx(-0.3, abs=0.05)
    assert challenger["bootstrap_ci_low"] < challenger["bootstrap_ci_high"]
    assert 0.0 <= float(challenger["bootstrap_pvalue"]) <= 1.0
    assert int(challenger["nw_lags"]) == 4
    assert bool(challenger["significant_vs_baseline"]) is True


def _synthetic_matrix() -> pd.DataFrame:
    """Synthetic feature matrix large enough for block length 10–15."""
    index = pd.bdate_range("2020-01-01", periods=N_SCREEN_ROWS)
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
        {
            "rv_cc_1d": rng.uniform(0.01, 0.05, N_SCREEN_ROWS),
            "rv_cc_5d": rng.uniform(0.02, 0.06, N_SCREEN_ROWS),
            "rv_cc_21d": rng.uniform(0.03, 0.07, N_SCREEN_ROWS),
            "ret_1d": rng.normal(0.0, 0.01, N_SCREEN_ROWS),
            "ret_5d": rng.normal(0.0, 0.02, N_SCREEN_ROWS),
            "range_1d": rng.uniform(0.005, 0.02, N_SCREEN_ROWS),
            "range_5d_mean": rng.uniform(0.005, 0.02, N_SCREEN_ROWS),
            "volume_z_21d": rng.normal(0.0, 1.0, N_SCREEN_ROWS),
        },
        index=index,
    )
    target = (
        0.05
        + 0.5 * features["rv_cc_1d"]
        + 0.3 * features["rv_cc_5d"]
        + 0.2 * features["rv_cc_21d"]
        + rng.normal(0.0, 0.001, N_SCREEN_ROWS)
    )
    features["target_rv_cc_5d"] = target
    return features


def test_screen_factors_writes_inference_artifacts(tmp_path: Path) -> None:
    """Screen should persist oos_losses / inference / enriched metrics."""

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
        inference=ScreenInferenceOptions(
            bootstrap=BootstrapInferenceOptions(
                block_length=10,
                n_resamples=99,
                alpha=0.05,
                random_seed=0,
            ),
            include_hln_dm=False,
            include_nonoverlap_sensitivity=True,
        ),
    )
    experiment_dir = (
        tmp_path / "artifacts" / result.identity.experiment_id.as_path_key()
    )
    for name in (
        "metrics.json",
        "oos_losses.json",
        "inference.json",
        "inference_sensitivity.json",
        "screen_meta.json",
    ):
        assert (experiment_dir / name).is_file()

    meta = artifact_store.read_json(result.identity.experiment_id, "screen_meta")
    assert meta["baseline_model"] == "har_rv_ols"
    assert meta["nw_lags"] == 4
    assert meta["bootstrap_block_length"] == 10
    assert meta["include_nonoverlap_sensitivity"] is True
    assert "mean_delta_qlike" in result.tables.summary.columns
    assert "bootstrap_pvalue" in result.tables.summary.columns
