"""Shared report methodology, caveats, and table helpers.

Exports
-------
InferenceReportMeta
    Inference fields shown in the locked-methodology list.
ReportMeta
    Locked methodology fields shown in the HTML memo.
format_oos_gap_wording
    Locked comparison text gated on primary bootstrap significance.
frame_records
    Convert a DataFrame to JSON-like row mappings (NaN → None).
null_if_missing
    Map pandas/NumPy missing values to None for Jinja.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vip.domain.errors import DataValidationError

DEFAULT_TARGET_COLUMN = "target_rv_cc_5d"
DEFAULT_PRIMARY_METRIC = "qlike"

DEFAULT_BASELINE_MODEL = "har_rv_ols"
DEFAULT_NW_LAGS = 4
DEFAULT_BOOTSTRAP_BLOCK_LENGTH = 15
DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_N_RESAMPLES = 1999

INFERENCE_CAVEATS = (
    "Horse-race QLIKE rankings without inference are descriptive, not findings.",
    "Overlapping multi-day RV labels induce dependence in loss differentials; "
    "primary inference is a block bootstrap of mean OOS ΔQLIKE vs HAR.",
    "Walk-forward embargo blocks train/test leakage; it is not a significance test.",
    "Say 'significantly better' only when the primary bootstrap rejects at alpha "
    "and mean ΔQLIKE favors the challenger; otherwise 'lower mean OOS QLIKE'.",
    "Non-overlapping every-horizon-day bootstrap is a footnote sensitivity check, "
    "not a second primary significance test.",
)
DEFAULT_CAVEATS = (
    "HAR lags (rv_cc_1d / rv_cc_5d / rv_cc_21d) are collinear; treat them as a "
    "feature family, not independent discoveries.",
    "Permutation importance is associative, not causal.",
    "Rankings can be unstable for weak factors and may shift across regimes.",
    "Results are for a single liquid ETF sample (SPY MVP) and should not be "
    "over-generalized.",
    "QLIKE permutation deltas can spike on collinear HAR lags; rankings use "
    "median importance across folds to limit single-fold domination.",
) + INFERENCE_CAVEATS


@dataclass(frozen=True, slots=True)
class InferenceReportMeta:
    """Inference fields shown in the locked-methodology list.

    Parameters
    ----------
    baseline_model : str
        Horse-race reference model.
    nw_lags : int
        Newey–West lag (horizon − 1).
    bootstrap_block_length : int
        Primary block-bootstrap length.
    alpha : float
        Two-sided significance level.
    bootstrap_n_resamples : int
        Bootstrap replication count.

    Methods
    -------
    validate()
        Raise if fields are invalid.
    describe()
        Return a short human-readable summary.
    """
    baseline_model: str = DEFAULT_BASELINE_MODEL
    nw_lags: int = DEFAULT_NW_LAGS
    bootstrap_block_length: int = DEFAULT_BOOTSTRAP_BLOCK_LENGTH
    alpha: float = DEFAULT_ALPHA
    bootstrap_n_resamples: int = DEFAULT_BOOTSTRAP_N_RESAMPLES

    def validate(self) -> None:
        """Raise ``DataValidationError`` when inference meta is invalid."""
        if not self.baseline_model.strip():
            raise DataValidationError("baseline_model must be non-empty.")
        if self.nw_lags < 0:
            raise DataValidationError("nw_lags must be non-negative.")
        if self.bootstrap_block_length < 1:
            raise DataValidationError("bootstrap_block_length must be >= 1.")
        if not 0.0 < self.alpha < 1.0:
            raise DataValidationError("alpha must be in (0, 1).")
        if self.bootstrap_n_resamples < 1:
            raise DataValidationError("bootstrap_n_resamples must be >= 1.")

    def describe(self) -> str:
        """Return a short human-readable summary."""
        return (
            f"baseline={self.baseline_model}, nw_lags={self.nw_lags}, "
            f"block={self.bootstrap_block_length}, alpha={self.alpha}"
        )


@dataclass(frozen=True, slots=True)
class ReportMeta:
    """Locked methodology fields for the factor-screen memo.

    Parameters
    ----------
    target_column : str
        Target column name.
    primary_metric : str
        Primary evaluation metric name.
    n_splits : int
        Number of walk-forward folds.
    embargo_size : int
        Embargo length in sessions.
    inference : InferenceReportMeta
        Bootstrap / NW / baseline fields for the methodology list.

    Methods
    -------
    validate()
        Raise if metadata is invalid.
    describe()
        Return a short human-readable summary.
    """

    target_column: str = DEFAULT_TARGET_COLUMN
    primary_metric: str = DEFAULT_PRIMARY_METRIC
    n_splits: int = 5
    embargo_size: int = 5
    inference: InferenceReportMeta = field(default_factory=InferenceReportMeta)

    def validate(self) -> None:
        """Raise ``DataValidationError`` when metadata is invalid."""
        if not self.target_column.strip():
            raise DataValidationError("target_column must be non-empty.")
        if not self.primary_metric.strip():
            raise DataValidationError("primary_metric must be non-empty.")
        if self.n_splits < 2:
            raise DataValidationError("n_splits must be at least 2.")
        if self.embargo_size < 0:
            raise DataValidationError("embargo_size must be non-negative.")
        self.inference.validate()

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact methodology summary.
        """
        return (
            f"{self.target_column} / {self.primary_metric} / "
            f"n_splits={self.n_splits} / embargo={self.embargo_size} / "
            f"{self.inference.describe()}"
        )


def format_oos_gap_wording(row: dict[str, object]) -> str:
    """Return locked comparison text for one horse-race row.

    Parameters
    ----------
    row : dict of str to object
        Summary row; may include inference columns.

    Returns
    -------
    str
        Baseline label, or gap wording gated on bootstrap significance.
    """
    mean_delta = row.get("mean_delta_qlike")
    if mean_delta is None or (isinstance(mean_delta, float) and pd.isna(mean_delta)):
        return "baseline (reference)"

    significant = bool(row.get("significant_vs_baseline"))
    delta = float(mean_delta)
    if significant and delta < 0.0:
        return "significantly lower mean OOS QLIKE vs HAR (bootstrap)"
    if delta < 0.0:
        return "lower mean OOS QLIKE vs HAR (not significant at α)"
    if delta > 0.0:
        return "higher mean OOS QLIKE vs HAR (not significant at α)"
    return "similar mean OOS QLIKE vs HAR"


def frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to JSON-like row mappings (NaN → None)."""
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        records.append({key: null_if_missing(value) for key, value in row.items()})
    return records


def null_if_missing(value: object) -> object:
    """Map pandas/NumPy missing values to None for Jinja."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return value
    return value
