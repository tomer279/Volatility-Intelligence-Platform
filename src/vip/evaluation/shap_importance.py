"""Walk-forward TreeSHAP importance for tree volatility models.

Exports
-------
shap_available
    Return whether the optional ``shap`` package can be imported.
shap_importance_folds
    Fit per fold, then rank features by mean |SHAP| on test rows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from vip.domain.errors import DataValidationError
from vip.evaluation.importance import WalkForwardSpec
from vip.evaluation.splitting import WalkForwardFold, generate_expanding_folds
from vip.evaluation.walk_forward import _align_features_and_target

ModelFactory = Callable[[], Any]


def shap_available() -> bool:
    """Return whether the optional ``shap`` dependency is installed.

    Returns
    -------
    bool
        True when ``import shap`` succeeds.
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        return False
    return True


def shap_importance_folds(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: ModelFactory,
    fold_spec: WalkForwardSpec,
) -> pd.DataFrame:
    """Compute fold-wise TreeSHAP importance on held-out test rows.

    For each expanding fold the model is fit on train only. A ``TreeExplainer``
    is built from the fitted tree estimator and applied to **test** features
    only. Importance is the mean absolute SHAP value per feature.

    Parameters
    ----------
    features : pandas.DataFrame
        Predictor matrix indexed by session date.
    target : pandas.Series
        Target aligned to ``features`` (used only for fitting).
    model_factory : callable
        Zero-arg factory returning a fresh tree model with
        ``fit`` / ``feature_names`` / ``fitted_estimator``.
    fold_spec : WalkForwardSpec
        Expanding walk-forward settings.

    Returns
    -------
    pandas.DataFrame
        Long-form table with columns:
        ``fold_id``, ``feature``, ``importance``.

    Raises
    ------
    DataValidationError
        If ``shap`` is missing, inputs are invalid, or no folds are scored.
    """
    shap_module = _import_shap()
    fold_spec.validate()

    feature_frame, target_series = _align_features_and_target(features, target)
    folds = generate_expanding_folds(
        index=feature_frame.index,
        n_splits=fold_spec.n_splits,
        embargo_size=fold_spec.embargo_size,
    )
    records = _shap_records_for_folds(
        feature_frame,
        target_series,
        model_factory,
        folds,
        shap_module,
    )
    if not records:
        raise DataValidationError("No SHAP-importance records were produced.")
    return pd.DataFrame.from_records(records)


def _import_shap() -> Any:
    """Import shap or raise a typed configuration error."""
    try:
        import shap
    except ImportError as exc:
        raise DataValidationError(
            "SHAP is not installed. Install with: pip install -e '.[nonlinear]'."
        ) from exc
    return shap


def _shap_records_for_folds(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: ModelFactory,
    folds: list[WalkForwardFold],
    shap_module: Any,
) -> list[dict[str, float | int | str]]:
    """Score SHAP importance on every fold."""
    records: list[dict[str, float | int | str]] = []
    for fold in folds:
        records.extend(
            _shap_records_for_fold(
                features,
                target,
                model_factory,
                fold,
                shap_module,
            )
        )
    return records


def _shap_records_for_fold(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: ModelFactory,
    fold: WalkForwardFold,
    shap_module: Any,
) -> list[dict[str, float | int | str]]:
    """Fit on train, explain test features, return per-feature records."""
    model = model_factory()
    model.fit(features.loc[fold.train_index], target.loc[fold.train_index])

    feature_names = list(model.feature_names())
    x_test = features.loc[fold.test_index, feature_names]
    explainer = shap_module.TreeExplainer(model.fitted_estimator())
    shap_values = explainer.shap_values(x_test.to_numpy(dtype=float))
    mean_abs = _mean_abs_shap(shap_values, n_features=len(feature_names))

    fold_id = int(fold.fold_id)
    return [
        {
            "fold_id": fold_id,
            "feature": name,
            "importance": float(mean_abs[position]),
        }
        for position, name in enumerate(feature_names)
    ]


def _mean_abs_shap(shap_values: Any, n_features: int) -> np.ndarray:
    """Reduce TreeExplainer output to mean |SHAP| per feature."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != n_features:
        raise DataValidationError(
            "Unexpected SHAP value shape for a regression tree model."
        )
    return np.mean(np.abs(values), axis=0)
