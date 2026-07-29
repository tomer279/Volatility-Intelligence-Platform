"""Tests for permutation-importance plotting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.visualization.importance_plots import (
    ImportancePlotOptions,
    plot_importance_bars,
)


def _toy_ranking() -> pd.DataFrame:
    """Build a tiny ranking table for plotting."""
    return pd.DataFrame(
        {
            "feature": ["rv_cc_5d", "rv_cc_1d", "noise"],
            "mean_importance": [0.40, 0.25, 0.01],
            "std_importance": [0.05, 0.04, 0.01],
            "top_k_hit_rate": [1.0, 0.6, 0.0],
            "n_folds": [3, 3, 3],
        }
    )


def test_plot_importance_bars_writes_png(tmp_path: Path) -> None:
    """Plot helper should write a non-empty PNG file."""
    output_path = tmp_path / "importance_plot.png"
    written = plot_importance_bars(_toy_ranking(), output_path)
    assert written == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_importance_bars_missing_columns_raise(tmp_path: Path) -> None:
    """Plot helper should require feature/mean_importance columns."""
    broken = _toy_ranking().drop(columns=["mean_importance"])
    with pytest.raises(DataValidationError, match="missing required columns"):
        plot_importance_bars(broken, tmp_path / "out.png")


def test_invalid_top_n_raises() -> None:
    """Plot options should reject non-positive top_n."""
    with pytest.raises(DataValidationError, match="top_n"):
        ImportancePlotOptions(top_n=0).validate()