"""Shared matplotlib style helpers for research plots.

Exports
-------
apply_research_style
    Apply a simple reproducible matplotlib style.
reset_research_style
    Restore matplotlib defaults after plotting.
"""

from __future__ import annotations

import matplotlib as mpl


def apply_research_style() -> None:
    """Apply a simple reproducible matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def reset_research_style() -> None:
    """Restore matplotlib default style parameters."""
    mpl.rcParams.update(mpl.rcParamsDefault)
