"""Walk-forward evaluation runner for volatility models.

Exports
-------
run_walk_forward
    Fit/predict/score models across expanding folds.
collect_walk_forward_predictions
    Collect dated out-of-sample predictions across expanding folds.
attach_qlike_losses
    Add per-row QLIKE losses to a walk-forward prediction panel.
collect_walk_forward_oos_losses
    Collect per-row OOS QLIKE losses across expanding folds.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.evaluation.metrics import (
    DEFAULT_EPSILON, mae, mse, qlike, qlike_losses
)
from vip.evaluation.splitting import WalkForwardFold, generate_expanding_folds

TARGET_COLUMN = "__target__"

OOS_LOSS_COLUMNS: tuple[str, ...] = (
    "model",
    "fold_id",
    "y_true",
    "y_pred",
    "qlike_loss",
)

class _SupportsFitPredict(Protocol):
    """Minimal model interface used by the walk-forward runner."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Any:
        """Fit on a training window."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict for a feature window."""


def run_walk_forward(
        features: pd.DataFrame,
        target: pd.Series,
        models: dict[str, _SupportsFitPredict],
        n_splits: int,
        embargo_size: int,
) -> pd.DataFrame:
    """Run walk-forward evaluation for one or more models.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature matrix indexed by session date.
    target : pandas.Series
        Target series aligned to ``features``.
    models : dict of str to model
        Mapping of model name to an object with ``fit`` / ``predict``.
        Each model is refit on every fold.
    n_splits : int
        Number of expanding test folds.
    embargo_size : int
        Embargo length in rows between train and test.

    Returns
    -------
    pandas.DataFrame
        Long-form metrics with columns:
        ``model``, ``fold_id``, ``qlike``, ``mse``, ``mae``,
        ``train_size``, ``test_size``.

    Raises
    ------
    DataValidationError
        If inputs are empty, misaligned, or no models are provided.
    """
    feature_frame, target_series = _align_features_and_target(features, target)
    if not models:
        raise DataValidationError("At least one model is required.")

    folds = generate_expanding_folds(
        index=feature_frame.index,
        n_splits=n_splits,
        embargo_size=embargo_size,
    )
    records = _score_models_on_folds(feature_frame, target_series, models, folds)
    return pd.DataFrame.from_records(records)


def _align_features_and_target(
        features: pd.DataFrame,
        target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align features and target on common finite rows.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature matrix.
    target : pandas.Series
        Target series.

    Returns
    -------
    tuple of pandas.DataFrame and pandas.Series
        Aligned features and target.

    Raises
    ------
    DataValidationError
        If inputs are empty or have no overlapping finite rows.
    """
    if features.empty:
        raise DataValidationError("Feature matrix must be non-empty.")

    aligned = features.join(target.rename(TARGET_COLUMN), how="inner").dropna()
    if aligned.empty:
        raise DataValidationError("No overlapping finite feature/target rows.")

    feature_frame = aligned.drop(columns=[TARGET_COLUMN])
    target_series = aligned[TARGET_COLUMN]
    return feature_frame, target_series


def _score_models_on_folds(
        features: pd.DataFrame,
        target: pd.Series,
        models: dict[str, _SupportsFitPredict],
        folds: list[WalkForwardFold],
) -> list[dict[str, float | int | str]]:
    """Score every model on every fold.

    Parameters
    ----------
    features : pandas.DataFrame
        Aligned feature matrix.
    target : pandas.Series
        Aligned target series.
    models : dict of str to model
        Models to evaluate.
    folds : list of WalkForwardFold
        Walk-forward folds.

    Returns
    -------
    list of dict
        Per-model, per-fold metric records.
    """
    records: list[dict[str, float | int | str]] = []
    for fold in folds:
        for model_name, model in models.items():
            record = _score_model_on_fold(features, target, model_name, model, fold)
            records.append(record)
    return records


def _score_model_on_fold(
        features: pd.DataFrame,
        target: pd.Series,
        model_name: str,
        model: _SupportsFitPredict,
        fold: WalkForwardFold,
) -> dict[str, float | int | str]:
    """Fit one model on a fold and return metric records.

    Parameters
    ----------
    features : pandas.DataFrame
        Aligned feature matrix.
    target : pandas.Series
        Aligned target series.
    model_name : str
        Model label used in output tables.
    model : _SupportsFitPredict
        Model instance to refit.
    fold : WalkForwardFold
        Train/test fold definition.

    Returns
    -------
    dict of str to float or int or str
        Metric record for this model/fold pair.
    """
    x_train = features.loc[fold.train_index]
    y_train = target.loc[fold.train_index]
    x_test = features.loc[fold.test_index]
    y_test = target.loc[fold.test_index]

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": model_name,
        "fold_id": int(fold.fold_id),
        "qlike": qlike(y_test, predictions),
        "mse": mse(y_test, predictions),
        "mae": mae(y_test, predictions),
        "train_size": int(fold.train_size()),
        "test_size": int(fold.test_size()),
    }


def collect_walk_forward_predictions(
        features: pd.DataFrame,
        target: pd.Series,
        models: dict[str, _SupportsFitPredict],
        n_splits: int,
        embargo_size: int,
) -> pd.DataFrame:
    """Collect out-of-sample predictions across expanding folds.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature matrix indexed by session date.
    target : pandas.Series
        Target series aligned to ``features``.
    models : dict of str to model
        Mapping of model name to ``fit`` / ``predict`` objects.
        Each model is refit on every fold.
    n_splits : int
        Number of expanding test folds.
    embargo_size : int
        Embargo length in rows between train and test.

    Returns
    -------
    pandas.DataFrame
        Long-form table with columns:
        ``model``, ``fold_id``, ``y_true``, ``y_pred``, indexed by
        session date (test rows only; may repeat dates across models).

    Raises
    ------
    DataValidationError
        If inputs are empty, misaligned, or no models are provided.
    """
    feature_frame, target_series = _align_features_and_target(features, target)
    if not models:
        raise DataValidationError("At least one model is required.")

    folds = generate_expanding_folds(
        index=feature_frame.index,
        n_splits=n_splits,
        embargo_size=embargo_size,
    )
    frames = [
        _predictions_for_model_on_folds(feature_frame, target_series, name, model, folds)
        for name, model in models.items()
    ]
    return pd.concat(frames, axis=0)


def _predictions_for_model_on_folds(
        features: pd.DataFrame,
        target: pd.Series,
        model_name: str,
        model: _SupportsFitPredict,
        folds: list[WalkForwardFold],
) -> pd.DataFrame:
    """Fit/predict one model on every fold and stack test rows."""
    pieces: list[pd.DataFrame] = []
    for fold in folds:
        x_train = features.loc[fold.train_index]
        y_train = target.loc[fold.train_index]
        x_test = features.loc[fold.test_index]
        y_test = target.loc[fold.test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        piece = pd.DataFrame(
            {
                "model": model_name,
                "fold_id": int(fold.fold_id),
                "y_true": y_test.to_numpy(dtype=float),
                "y_pred": y_pred.to_numpy(dtype=float),
            },
            index=y_test.index,
        )
        pieces.append(piece)
    return pd.concat(pieces, axis=0)


def attach_qlike_losses(
        predictions: pd.DataFrame,
        epsilon: float = DEFAULT_EPSILON,
) -> pd.DataFrame:
    """Add per-row QLIKE losses to a walk-forward prediction panel.

    Rows are treated as already paired. This supports stacked multi-model
    panels whose date index repeats across models.
    """
    required = {"model", "fold_id", "y_true", "y_pred"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"predictions missing required columns: {missing_text}."
        )
    if predictions.empty:
        raise DataValidationError("predictions must be non-empty.")
    if epsilon <= 0:
        raise DataValidationError("epsilon must be positive.")

    frame = predictions.copy()
    # Positional pairing: avoid index align (duplicate dates across models).
    y_true = pd.Series(frame["y_true"].to_numpy(dtype=float))
    y_pred = pd.Series(frame["y_pred"].to_numpy(dtype=float))
    frame["qlike_loss"] = qlike_losses(y_true, y_pred, epsilon=epsilon).to_numpy(
        dtype=float
    )
    return frame


def collect_walk_forward_oos_losses(
        features: pd.DataFrame,
        target: pd.Series,
        models: dict[str, _SupportsFitPredict],
        n_splits: int,
        embargo_size: int,
) -> pd.DataFrame:
    """Collect per-row OOS QLIKE losses across expanding folds.
    Shares fits with ``collect_walk_forward_predictions``: one walk-forward
    pass, then elementwise QLIKE.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature matrix indexed by session date.
    target : pandas.Series
        Target series aligned to ``features``.
    models : dict of str to model
        Mapping of model name to ``fit`` / ``predict`` objects.
    n_splits : int
        Number of expanding test folds.
    embargo_size : int
        Embargo length in rows between train and test.

    Returns
    -------
    pandas.DataFrame
        Long-form panel with columns ``model``, ``fold_id``, ``y_true``,
        ``y_pred``, ``qlike_loss``, indexed by session date.

    Raises
    ------
    DataValidationError
        If inputs are empty, misaligned, or no models are provided.
    """
    predictions = collect_walk_forward_predictions(
        features=features,
        target=target,
        models=models,
        n_splits=n_splits,
        embargo_size=embargo_size,
    )
    return attach_qlike_losses(predictions)
