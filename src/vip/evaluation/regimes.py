"""Named evaluation regimes and sliced forecast metrics.

Exports
-------
RegimeWindow
    Inclusive calendar window with a stable research name.
locked_regime_windows
    Return the Milestone 5 COVID / 2022 windows.
score_predictions_by_regime
    Score OOS predictions for full sample and each locked regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.evaluation.metrics import mae, mse, qlike

FULL_SAMPLE_NAME = "full_sample"
REQUIRED_PRED_COLUMNS = ("model", "y_true", "y_pred")


@dataclass(frozen=True, slots=True)
class RegimeWindow:
    """Inclusive calendar window for regime-sliced evaluation.

    Parameters
    ----------
    name : str
        Stable regime identifier.
    start : datetime.date
        First calendar day in the window (inclusive).
    end : datetime.date
        Last calendar day in the window (inclusive).

    Methods
    -------
    mask(index)
        Boolean mask of index entries inside this window.
    describe()
        Return a short human-readable summary.
    """

    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise DataValidationError("Regime name must be non-empty.")
        if self.end < self.start:
            raise DataValidationError(
                f"Regime '{self.name}' has end before start."
            )

    def mask(self, index: pd.DatetimeIndex) -> pd.Series:
        """Return a boolean mask for dates inside this window.

        Parameters
        ----------
        index : pandas.DatetimeIndex
            Session dates to test.

        Returns
        -------
        pandas.Series
            Boolean series aligned to ``index``.
        """
        sessions = pd.Series(index.date, index=index)
        return (sessions >= self.start) & (sessions <= self.end)

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact regime description.
        """
        return f"{self.name}: {self.start.isoformat()}..{self.end.isoformat()}"


def locked_regime_windows() -> tuple[RegimeWindow, ...]:
    """Return the locked Milestone 5 stress windows.

    Returns
    -------
    tuple of RegimeWindow
        COVID crash and 2022 bear windows (full sample is handled separately).
    """
    return (
        RegimeWindow(
            name="covid_crash",
            start=date(2020, 2, 20),
            end=date(2020, 4, 30),
        ),
        RegimeWindow(
            name="bear_2022",
            start=date(2022, 1, 3),
            end=date(2022, 10, 14),
        ),
    )


def score_predictions_by_regime(
    predictions: pd.DataFrame,
    regimes: tuple[RegimeWindow, ...] | None = None,
) -> pd.DataFrame:
    """Score walk-forward OOS predictions by regime and model.

    Always includes a ``full_sample`` block. Regime blocks with no overlapping
    rows are included with ``n_obs=0`` and null metrics (report-friendly).

    Parameters
    ----------
    predictions : pandas.DataFrame
        Output of ``collect_walk_forward_predictions`` with columns
        ``model``, ``y_true``, ``y_pred`` and a DatetimeIndex.
    regimes : tuple of RegimeWindow or None, default None
        Windows to score. ``None`` uses ``locked_regime_windows()``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``regime``, ``model``, ``n_obs``, ``qlike``, ``mse``, ``mae``.

    Raises
    ------
    DataValidationError
        If ``predictions`` is empty or missing required columns.
    """
    _validate_predictions(predictions)
    windows = regimes if regimes is not None else locked_regime_windows()

    records: list[dict[str, float | int | str | None]] = []
    records.extend(_score_regime_block(predictions, FULL_SAMPLE_NAME, None))
    for window in windows:
        records.extend(_score_regime_block(predictions, window.name, window))
    return pd.DataFrame.from_records(records)


def _validate_predictions(predictions: pd.DataFrame) -> None:
    """Validate prediction-frame schema."""
    if predictions.empty:
        raise DataValidationError("predictions frame must be non-empty.")
    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise DataValidationError("predictions index must be a DatetimeIndex.")
    missing = [c for c in REQUIRED_PRED_COLUMNS if c not in predictions.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"predictions missing required columns: {missing_text}."
        )


def _score_regime_block(
    predictions: pd.DataFrame,
    regime_name: str,
    window: RegimeWindow | None,
) -> list[dict[str, float | int | str | None]]:
    """Score every model inside one regime (or full sample)."""
    if window is None:
        slice_frame = predictions
    else:
        slice_frame = predictions.loc[window.mask(predictions.index)]

    records: list[dict[str, float | int | str | None]] = []
    for model_name, group in slice_frame.groupby("model", sort=False):
        records.append(_metric_record(regime_name, str(model_name), group))
    # Ensure models present in full predictions still appear when slice is empty.
    if slice_frame.empty:
        for model_name in predictions["model"].unique():
            records.append(_empty_metric_record(regime_name, str(model_name)))
    return records


def _metric_record(
    regime_name: str,
    model_name: str,
    group: pd.DataFrame,
) -> dict[str, float | int | str | None]:
    """Build one regime/model metric row."""
    if group.empty:
        return _empty_metric_record(regime_name, model_name)
    y_true = group["y_true"]
    y_pred = group["y_pred"]
    return {
        "regime": regime_name,
        "model": model_name,
        "n_obs": int(len(group)),
        "qlike": float(qlike(y_true, y_pred)),
        "mse": float(mse(y_true, y_pred)),
        "mae": float(mae(y_true, y_pred)),
    }


def _empty_metric_record(
    regime_name: str,
    model_name: str,
) -> dict[str, float | int | str | None]:
    """Build a placeholder row for an empty regime slice."""
    return {
        "regime": regime_name,
        "model": model_name,
        "n_obs": 0,
        "qlike": None,
        "mse": None,
        "mae": None,
    }
