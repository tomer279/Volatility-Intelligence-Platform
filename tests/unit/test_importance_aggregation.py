"""Tests for median ranking and optional ΔQLIKE capping."""

from __future__ import annotations

import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.evaluation.importance import ImportanceOptions
from vip.evaluation.stability import StabilityOptions, summarize_importance


def _spiked_importance() -> pd.DataFrame:
    """Signal wins on median; one noise spike inflates the mean."""
    return pd.DataFrame(
        {
            "fold_id": [0, 0, 1, 1, 2, 2],
            "feature": ["signal", "noise", "signal", "noise", "signal", "noise"],
            "importance": [0.50, 0.05, 0.40, 0.02, 0.45, 100.0],
            "baseline_qlike": [1.0] * 6,
            "n_repeats": [3] * 6,
        }
    )


def test_median_rank_keeps_signal_first_despite_spike() -> None:
    """A single-fold noise spike must not beat signal under median rank."""
    summary = summarize_importance(
        _spiked_importance(),
        options=StabilityOptions(top_k=1, rank_by="median"),
    )
    assert list(summary["feature"]) == ["signal", "noise"]
    assert float(summary.iloc[0]["median_importance"]) > float(
        summary.iloc[1]["median_importance"]
    )


def test_mean_rank_can_prefer_spiked_noise() -> None:
    """Mean aggregation may let a spike dominate (documents why median is default)."""
    summary = summarize_importance(
        _spiked_importance(),
        options=StabilityOptions(top_k=1, rank_by="mean"),
    )
    assert list(summary["feature"]) == ["noise", "signal"]


def test_summary_includes_mean_and_median_columns() -> None:
    """Summary should always report both aggregates."""
    summary = summarize_importance(_spiked_importance())
    assert "mean_importance" in summary.columns
    assert "median_importance" in summary.columns


def test_invalid_delta_cap_raises() -> None:
    """Non-positive delta_cap should be rejected."""
    with pytest.raises(DataValidationError, match="delta_cap"):
        ImportanceOptions(delta_cap=0.0).validate()