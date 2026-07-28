"""Helpers for summarizing walk-forward metric tables.

Exports
-------
summarize_walk_forward
    Aggregate fold metrics by model and sort by a primary metric.
"""

from __future__ import annotations

import pandas as pd

from vip.domain.errors import DataValidationError

PRIMARY_METRIC = "qlike"
METRIC_COLUMNS: tuple[str, ...] = ("qlike", "mse", "mae")


def summarize_walk_forward(
    fold_metrics: pd.DataFrame,
    primary_metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Aggregate walk-forward fold metrics by model.

    Parameters
    ----------
    fold_metrics : pandas.DataFrame
        Output of ``run_walk_forward``.
    primary_metric : str, default ``'qlike'``
        Metric used for ascending sort (lower is better).

    Returns
    -------
    pandas.DataFrame
        One row per model with mean metric values, sorted by
        ``primary_metric`` ascending.

    Raises
    ------
    DataValidationError
        If required columns are missing or ``primary_metric`` is unknown.
    """
    required = {"model", *METRIC_COLUMNS}
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"fold_metrics missing required columns: {missing_text}."
        )
    if primary_metric not in METRIC_COLUMNS:
        raise DataValidationError(
            f"Unsupported primary metric '{primary_metric}'."
        )

    summary = (
        fold_metrics.groupby("model", sort=False)[list(METRIC_COLUMNS)]
        .mean()
        .reset_index()
        .sort_values(primary_metric, ascending=True)
        .reset_index(drop=True)
    )
    return summary
