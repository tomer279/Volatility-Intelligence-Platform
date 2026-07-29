"""Permutation-importance plotting helpers.

Exports
-------
ImportancePlotOptions
    Display options for importance bar charts.
plot_importance_bars
    Write a horizontal bar chart of mean feature importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vip.domain.errors import DataValidationError
from vip.visualization.styles import apply_research_style, reset_research_style

REQUIRED_COLUMNS = ("feature", "mean_importance")
REQUIRED_FEATURE_COLUMN = "feature"
IMPORTANCE_COLUMN_CANDIDATES = ("median_importance", "mean_importance")
DEFAULT_TITLE = "Permutation importance (median ΔQLIKE)"
DEFAULT_TOP_N = 20
DEFAULT_FIGURE_WIDTH = 8.0
DEFAULT_FIGURE_HEIGHT = 4.5


@dataclass(frozen=True, slots=True)
class ImportancePlotOptions:
    """Display options for an importance bar chart.

    Parameters
    ----------
    title : str, default DEFAULT_TITLE
        Figure title.
    top_n : int, default 20
        Maximum number of features to display.
    figure_width : float, default 8.0
        Figure width in inches.
    figure_height : float, default 4.5
        Figure height in inches.

    Methods
    -------
    validate()
        Raise if options are invalid.
    describe()
        Return a short human-readable summary.
    """

    title: str = DEFAULT_TITLE
    top_n: int = DEFAULT_TOP_N
    figure_width: float = DEFAULT_FIGURE_WIDTH
    figure_height: float = DEFAULT_FIGURE_HEIGHT

    def validate(self) -> None:
        """Raise ``DataValidationError`` when options are invalid."""
        if self.top_n < 1:
            raise DataValidationError("top_n must be at least 1.")
        if self.figure_width <= 0 or self.figure_height <= 0:
            raise DataValidationError("figure dimensions must be positive.")

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact description of plot options.
        """
        return f"top_n={self.top_n}, title={self.title!r}"


def plot_importance_bars(
    ranking: pd.DataFrame,
    output_path: Path,
    options: ImportancePlotOptions | None = None,
) -> Path:
    """Write a horizontal bar chart of mean feature importance.

    Parameters
    ----------
    ranking : pandas.DataFrame
        Stability summary with ``feature`` and ``mean_importance``.
    output_path : pathlib.Path
        Destination PNG path.
    options : ImportancePlotOptions or None, default None
        Display options.

    Returns
    -------
    pathlib.Path
        Path written to disk.

    Raises
    ------
    DataValidationError
        If ``ranking`` is empty/invalid or the figure cannot be written.
    """
    resolved = options if options is not None else ImportancePlotOptions()
    resolved.validate()
    frame = _prepare_ranking(ranking, resolved.top_n)

    apply_research_style()
    try:
        _render_and_save(frame, output_path, resolved)
    finally:
        reset_research_style()
        plt.close("all")
    return output_path


def _resolve_importance_column(ranking: pd.DataFrame) -> str:
    """Return the preferred importance column present in ``ranking``."""
    for column in IMPORTANCE_COLUMN_CANDIDATES:
        if column in ranking.columns:
            return column
    raise DataValidationError(
        "ranking table missing importance columns: "
        "median_importance or mean_importance."
    )


def _prepare_ranking(ranking: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Validate and select the top features for plotting."""
    if ranking.empty:
        raise DataValidationError("ranking table must be non-empty.")
    if REQUIRED_FEATURE_COLUMN not in ranking.columns:
        raise DataValidationError("ranking table missing required column: feature.")
    importance_column = _resolve_importance_column(ranking)
    ordered = ranking.sort_values(
        by=importance_column,
        ascending=False,
        kind="mergesort",
    ).head(top_n)
    frame = ordered.iloc[::-1].copy()
    frame["__plot_importance__"] = frame[importance_column]
    return frame


def _render_and_save(
    frame: pd.DataFrame,
    output_path: Path,
    options: ImportancePlotOptions,
) -> None:
    """Render the bar chart and save it as PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        figsize=(options.figure_width, options.figure_height),
    )
    axes.barh(
        frame["feature"],
        frame["__plot_importance__"],
        color="#4C78A8",
    )
    axes.set_xlabel("Importance")
    axes.set_ylabel("Feature")
    axes.set_title(options.title)
    figure.tight_layout()
    try:
        figure.savefig(output_path, dpi=120)
    except OSError as exc:
        raise DataValidationError(
            f"Failed to write importance plot to {output_path}: {exc}"
        ) from exc
