"""Tree-based volatility models (unscaled features).

Exports
-------
TreeVolModel
    Adapter around a scikit-learn-style tree regressor.
RandomForestVolModel
    Random-forest volatility forecast adapter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor

from vip.domain.errors import DataValidationError

DEFAULT_PREDICTION_FLOOR = 1e-8
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 4
DEFAULT_MIN_SAMPLES_LEAF = 5
DEFAULT_RANDOM_STATE = 0


class TreeVolModel:
    """Tree regressor fit on raw (unscaled) features.

    Estimator fitting happens only inside ``fit`` (typically on a training
    fold). Predictions reuse that train-time estimator.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Unfitted tree-style regressor (cloned on each ``fit``).
    prediction_floor : float, default 1e-8
        Lower bound applied to predictions.

    Methods
    -------
    fit(features, target)
        Fit the estimator on finite training rows.
    predict(features)
        Predict with the fitted estimator and apply the floor.
    feature_names()
        Return the feature names used at fit time.
    feature_importances()
        Return impurity-based importances keyed by feature name.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    ) -> None:
        """Initialize an unfitted tree volatility model.

        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            Unfitted regressor template.
        prediction_floor : float, default 1e-8
            Minimum allowed prediction.

        Raises
        ------
        DataValidationError
            If ``prediction_floor`` is not positive.
        """
        if prediction_floor <= 0:
            raise DataValidationError("prediction_floor must be positive.")
        self._base_estimator = estimator
        self._prediction_floor = prediction_floor
        self._estimator: BaseEstimator | None = None
        self._feature_names: tuple[str, ...] | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> TreeVolModel:
        """Fit the tree estimator on aligned finite rows.

        Parameters
        ----------
        features : pandas.DataFrame
            Training feature matrix.
        target : pandas.Series
            Training realized-volatility target.

        Returns
        -------
        TreeVolModel
            Fitted model (``self``).

        Raises
        ------
        DataValidationError
            If ``features`` is empty or no finite rows remain.
        """
        if features.empty:
            raise DataValidationError("Feature matrix must be non-empty.")

        frame = features.copy()
        frame["__target__"] = target
        clean = frame.dropna()
        if clean.empty:
            raise DataValidationError("No finite rows available to fit TreeVolModel.")

        feature_frame = clean.drop(columns=["__target__"])
        clean_target = clean["__target__"]
        feature_names = tuple(str(column) for column in feature_frame.columns)

        estimator = clone(self._base_estimator)
        estimator.fit(
            feature_frame.to_numpy(dtype=float),
            clean_target.to_numpy(dtype=float),
        )

        self._estimator = estimator
        self._feature_names = feature_names
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict realized volatility for ``features``.

        Parameters
        ----------
        features : pandas.DataFrame
            Feature rows to score (same columns as training).

        Returns
        -------
        pandas.Series
            Predictions aligned to ``features.index``, floored at
            ``prediction_floor``.

        Raises
        ------
        DataValidationError
            If the model is unfitted or required columns are missing.
        """
        estimator, feature_names = self._require_fitted_state()
        missing = [name for name in feature_names if name not in features.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise DataValidationError(
                f"Missing feature columns for prediction: {missing_text}."
            )

        ordered = features.loc[:, list(feature_names)]
        raw = estimator.predict(ordered.to_numpy(dtype=float))
        clipped = np.maximum(np.asarray(raw, dtype=float), self._prediction_floor)
        return pd.Series(clipped, index=features.index, name="prediction")

    def feature_names(self) -> tuple[str, ...]:
        """Return feature names captured at fit time.

        Returns
        -------
        tuple of str
            Training feature column names.

        Raises
        ------
        DataValidationError
            If the model is unfitted.
        """
        _, feature_names = self._require_fitted_state()
        return feature_names

    def feature_importances(self) -> pd.Series:
        """Return impurity-based importances keyed by feature name.

        Returns
        -------
        pandas.Series
            Non-negative importances in training column order.

        Raises
        ------
        DataValidationError
            If the model is unfitted or the estimator has no importances.
        """
        estimator, feature_names = self._require_fitted_state()
        if not hasattr(estimator, "feature_importances_"):
            raise DataValidationError(
                "Fitted estimator does not expose feature_importances_."
            )
        values = np.asarray(estimator.feature_importances_, dtype=float).ravel()
        return pd.Series(values, index=list(feature_names), name="importance")

    def _require_fitted_state(self) -> tuple[BaseEstimator, tuple[str, ...]]:
        """Return fitted estimator/names or raise."""
        if self._estimator is None or self._feature_names is None:
            raise DataValidationError("TreeVolModel must be fitted before use.")
        return self._estimator, self._feature_names

    def fitted_estimator(self) -> BaseEstimator:
        """Return the fitted scikit-learn estimator.

        Returns
        -------
        sklearn.base.BaseEstimator
            Estimator produced by the last ``fit`` call.

        Raises
        ------
        DataValidationError
            If the model is unfitted.
        """
        estimator, _ = self._require_fitted_state()
        return estimator

class RandomForestVolModel(TreeVolModel):
    """Random-forest volatility model on unscaled features.

    Parameters
    ----------
    n_estimators : int, default 200
        Number of trees.
    max_depth : int, default 4
        Maximum tree depth.
    min_samples_leaf : int, default 5
        Minimum samples required at a leaf node.
    random_state : int, default 0
        RNG seed for reproducibility.
    prediction_floor : float, default 1e-8
        Lower bound applied to predictions.

    Methods
    -------
    fit(features, target)
        Fit the forest on training data.
    predict(features)
        Predict with the fitted forest.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        random_state: int = DEFAULT_RANDOM_STATE,
        prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    ) -> None:
        """Initialize an unfitted random-forest model.

        Parameters
        ----------
        n_estimators : int, default 200
            Number of trees.
        max_depth : int, default 4
            Maximum tree depth.
        min_samples_leaf : int, default 5
            Minimum samples at a leaf.
        random_state : int, default 0
            RNG seed.
        prediction_floor : float, default 1e-8
            Minimum allowed prediction.

        Raises
        ------
        DataValidationError
            If hyperparameters are invalid.
        """
        if n_estimators < 1:
            raise DataValidationError("n_estimators must be at least 1.")
        if max_depth < 1:
            raise DataValidationError("max_depth must be at least 1.")
        if min_samples_leaf < 1:
            raise DataValidationError("min_samples_leaf must be at least 1.")
        super().__init__(
            estimator=RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=1,
            ),
            prediction_floor=prediction_floor,
        )
