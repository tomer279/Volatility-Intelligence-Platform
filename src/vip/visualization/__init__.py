"""Plotting helpers for research diagnostics and reports.

Exports
-------
ImportancePlotOptions
    Display options for importance bar charts.
plot_importance_bars
    Write a horizontal bar chart of mean feature importance.
apply_research_style
    Apply a simple reproducible matplotlib style.
reset_research_style
    Restore matplotlib defaults after plotting.
"""

from vip.visualization.importance_plots import (
    ImportancePlotOptions,
    plot_importance_bars,
)
from vip.visualization.styles import apply_research_style, reset_research_style

__all__ = [
    "ImportancePlotOptions",
    "apply_research_style",
    "plot_importance_bars",
    "reset_research_style",
]