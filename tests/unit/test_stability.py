"""Tests for factor importance stability summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.evaluation.stability import StabilityOptions, summarize_importance


def _toy_importance() -> pd.DataFrame:
    """Build a tiny fold-wise importance table."""
    return pd.DataFrame(
        {
            "fold_id": [0, 0, 1, 1, 2, 2],
            "feature": ["signal", "noise", "signal", "noise", "signal", "noise"],
            "importance": [0.50, 0.05, 0.40, 0.02, 0.45, 0.10],
            "baseline_qlike": [1.0] * 6,
            "n_repeats": [3] * 6,
        }
    )


def test_summarize_importance_ranks_signal_first() -> None:
    """Higher median-importance features should appear first by default."""
    summary = summarize_importance(_toy_importance())
    assert list(summary["feature"]) == ["signal", "noise"]
    assert "median_importance" in summary.columns
    assert float(summary.iloc[0]["median_importance"]) > float(
        summary.iloc[1]["median_importance"]
    )


def test_top_k_hit_rate() -> None:
    """Signal should be top-1 in every fold on the toy table."""
    summary = summarize_importance(
        _toy_importance(),
        options=StabilityOptions(top_k=1),
    )
    signal_row = summary.loc[summary["feature"] == "signal"].iloc[0]
    noise_row = summary.loc[summary["feature"] == "noise"].iloc[0]
    assert float(signal_row["top_k_hit_rate"]) == pytest.approx(1.0)
    assert float(noise_row["top_k_hit_rate"]) == pytest.approx(0.0)


def test_missing_columns_raise() -> None:
    """Summary should require fold_id/feature/importance columns."""
    broken = _toy_importance().drop(columns=["importance"])
    with pytest.raises(DataValidationError, match="missing required columns"):
        summarize_importance(broken)