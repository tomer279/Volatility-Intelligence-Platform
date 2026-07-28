"""Tests for walk-forward evaluation and comparison summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vip.evaluation.comparison import summarize_walk_forward
from vip.evaluation.walk_forward import run_walk_forward
from vip.modeling.baselines import HarRvOlsModel, HistoricalMeanModel


def _synthetic_design(n_rows: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    """Build a synthetic design where HAR should beat the mean."""
    index = pd.bdate_range("2020-01-01", periods=n_rows)
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            "rv_cc_1d": rng.uniform(0.01, 0.05, n_rows),
            "rv_cc_5d": rng.uniform(0.02, 0.06, n_rows),
            "rv_cc_21d": rng.uniform(0.03, 0.07, n_rows),
        },
        index=index,
    )
    target = (
        0.05
        + 0.5 * features["rv_cc_1d"]
        + 0.3 * features["rv_cc_5d"]
        + 0.2 * features["rv_cc_21d"]
        + rng.normal(0.0, 0.001, n_rows)
    )
    return features, pd.Series(target, index=index, name="target_rv_cc_5d")


def test_run_walk_forward_returns_expected_schema() -> None:
    """Walk-forward output should include per-model fold metrics."""
    features, target = _synthetic_design()
    fold_metrics = run_walk_forward(
        features=features,
        target=target,
        models={
            "historical_mean": HistoricalMeanModel(),
            "har_rv_ols": HarRvOlsModel(),
        },
        n_splits=4,
        embargo_size=5,
    )

    assert set(fold_metrics.columns) == {
        "model",
        "fold_id",
        "qlike",
        "mse",
        "mae",
        "train_size",
        "test_size",
    }
    assert set(fold_metrics["model"]) == {"historical_mean", "har_rv_ols"}
    assert fold_metrics["fold_id"].nunique() == 4


def test_har_beats_mean_on_synthetic_qlike() -> None:
    """On a HAR-like synthetic target, HAR OLS should beat the mean on QLIKE."""
    features, target = _synthetic_design()
    fold_metrics = run_walk_forward(
        features=features,
        target=target,
        models={
            "historical_mean": HistoricalMeanModel(),
            "har_rv_ols": HarRvOlsModel(),
        },
        n_splits=4,
        embargo_size=5,
    )
    summary = summarize_walk_forward(fold_metrics, primary_metric="qlike")
    assert summary.iloc[0]["model"] == "har_rv_ols"
    har_qlike = float(summary.loc[summary["model"] == "har_rv_ols", "qlike"].iloc[0])
    mean_qlike = float(
        summary.loc[summary["model"] == "historical_mean", "qlike"].iloc[0]
    )
    assert har_qlike < mean_qlike