"""Walk-forward evaluation runner for volatility models.

Exports
-------
run_walk_forward
    Fit/predict/score models across expanding folds.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.evaluation.metrics import mae, mse, qlike
from vip.evaluation.splitting import WalkForwardFold, generate_expanding_folds

TARGET_COLUMN = "__target__"


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
