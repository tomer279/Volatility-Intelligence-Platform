"""Factor-importance stability summaries across walk-forward folds.

Exports
-------
StabilityOptions
    Top-k settings for hit-rate calculations.
summarize_importance
    Aggregate fold-wise importance into a ranked factor table.
top_k_hit_rate
    Share of folds where each feature ranks in the top-k.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vip.domain.errors import DataValidationError

DEFAULT_TOP_K = 3
REQUIRED_COLUMNS = ("fold_id", "feature", "importance")


@dataclass(frozen=True, slots=True)
class StabilityOptions:
    """Options for importance stability summaries.

    Parameters
    ----------
    top_k : int, default 3
        Rank threshold used for top-k hit-rate.

    Methods
    -------
    validate()
        Raise if options are invalid.
    describe()
        Return a short human-readable summary.
    """

    top_k: int = DEFAULT_TOP_K

    def validate(self) -> None:
        """Raise ``DataValidationError`` when options are invalid."""
        if self.top_k < 1:
            raise DataValidationError("top_k must be at least 1.")

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact description of stability options.
        """
        return f"top_k={self.top_k}"


def summarize_importance(
    importance: pd.DataFrame,
    options: StabilityOptions | None = None,
) -> pd.DataFrame:
    """Aggregate fold-wise importance into a ranked factor table.

    Parameters
    ----------
    importance : pandas.DataFrame
        Long-form output from ``permutation_importance_folds`` with
        columns ``fold_id``, ``feature``, ``importance``.
    options : StabilityOptions or None, default None
        Top-k settings (defaults used when ``None``).

    Returns
    -------
    pandas.DataFrame
        Ranked table with columns:
        ``feature``, ``mean_importance``, ``std_importance``,
        ``n_folds``, ``top_k_hit_rate``, sorted by mean importance
        descending.

    Raises
    ------
    DataValidationError
        If ``importance`` is empty or missing required columns.
    """
    resolved = options if options is not None else StabilityOptions()
    resolved.validate()
    _validate_importance_frame(importance)

    grouped = importance.groupby("feature", sort=False)["importance"]
    summary = grouped.agg(mean_importance="mean", std_importance="std", n_folds="count")
    summary = summary.reset_index()
    summary["std_importance"] = summary["std_importance"].fillna(0.0)

    hit_rates = top_k_hit_rate(importance, resolved)
    summary = summary.merge(hit_rates, on="feature", how="left")
    summary["top_k_hit_rate"] = summary["top_k_hit_rate"].fillna(0.0)
    summary = summary.sort_values(
        by=["mean_importance", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return summary


def top_k_hit_rate(
    importance: pd.DataFrame,
    options: StabilityOptions | None = None,
) -> pd.DataFrame:
    """Compute the share of folds where each feature is in the top-k.

    Parameters
    ----------
    importance : pandas.DataFrame
        Long-form fold/feature importance table.
    options : StabilityOptions or None, default None
        Top-k settings (defaults used when ``None``).

    Returns
    -------
    pandas.DataFrame
        Columns ``feature``, ``top_k_hit_rate`` with rates in ``[0, 1]``.

    Raises
    ------
    DataValidationError
        If ``importance`` is empty or missing required columns.
    """
    resolved = options if options is not None else StabilityOptions()
    resolved.validate()
    _validate_importance_frame(importance)

    ranked = importance.copy()
    ranked["rank"] = ranked.groupby("fold_id")["importance"].rank(
        method="first",
        ascending=False,
    )
    hits = ranked.loc[ranked["rank"] <= resolved.top_k]
    fold_count = int(importance["fold_id"].nunique())
    hit_counts = hits.groupby("feature").size()
    all_features = pd.Index(importance["feature"].unique(), name="feature")
    rates = (hit_counts.reindex(all_features, fill_value=0) / fold_count).astype(float)
    return rates.rename("top_k_hit_rate").reset_index()


def _validate_importance_frame(importance: pd.DataFrame) -> None:
    """Validate the long-form importance schema."""
    if importance.empty:
        raise DataValidationError("Importance frame must be non-empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in importance.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"Importance frame missing required columns: {missing_text}."
        )
