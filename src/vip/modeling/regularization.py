"""Regularized linear volatility models with train-only scaling.

Exports
-------
ScaledLinearModel
    StandardScaler plus a scikit-learn linear estimator.
RidgeModel
    Ridge regression adapter.
LassoModel
    Lasso regression adapter.
ElasticNetModel
    Elastic-net regression adapter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from vip.domain.errors import DataValidationError

DEFAULT_PREDICTION_FLOOR = 1e-8
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_LASSO_ALPHA = 0.001
DEFAULT_ELASTICNET_ALPHA = 0.001
DEFAULT_ELASTICNET_L1_RATIO = 0.5
DEFAULT_MAX_ITER = 10_000


class ScaledLinearModel:
    """Linear model fit on standard-scaled features.

    The scaler and estimator are fit only inside ``fit`` (typically on a
    training fold). Predictions reuse that train-time scaler.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Unfitted scikit-learn regressor (cloned on each ``fit``).
    prediction_floor : float, default 1e-8
        Lower bound applied to predictions.

    Methods
    -------
    fit(features, target)
        Fit scaler and estimator on finite training rows.
    predict(features)
        Predict with the train-time scaler and fitted estimator.
    coefficients()
        Return the fitted linear coefficients keyed by feature name.
    feature_names()
        Return the feature names used at fit time.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    ) -> None:
        """Initialize an unfitted scaled linear model.

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
        self._scaler: StandardScaler | None = None
        self._estimator: BaseEstimator | None = None
        self._feature_names: tuple[str, ...] | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ScaledLinearModel:
        """Fit scaler and estimator on aligned finite rows.

        Parameters
        ----------
        features : pandas.DataFrame
            Training feature matrix.
        target : pandas.Series
            Training realized-volatility target.

        Returns
        -------
        ScaledLinearModel
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
            raise DataValidationError("No finite rows available to fit ScaledLinearModel.")

        feature_frame = clean.drop(columns=["__target__"])
        clean_target = clean["__target__"]
        feature_names = tuple(str(column) for column in feature_frame.columns)

        scaler = StandardScaler()
        design = scaler.fit_transform(feature_frame.to_numpy(dtype=float))
        estimator = clone(self._base_estimator)
        estimator.fit(design, clean_target.to_numpy(dtype=float))

        self._scaler = scaler
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
        scaler, estimator, feature_names = self._require_fitted_state()
        missing = [name for name in feature_names if name not in features.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise DataValidationError(
                f"Missing feature columns for prediction: {missing_text}."
            )

        ordered = features.loc[:, list(feature_names)]
        design = scaler.transform(ordered.to_numpy(dtype=float))
        raw = estimator.predict(design)
        clipped = np.maximum(np.asarray(raw, dtype=float), self._prediction_floor)
        return pd.Series(clipped, index=features.index, name="prediction")

    def coefficients(self) -> pd.Series:
        """Return fitted coefficients keyed by feature name.

        Returns
        -------
        pandas.Series
            Linear coefficients in training column order.

        Raises
        ------
        DataValidationError
            If the model is unfitted.
        """
        _, estimator, feature_names = self._require_fitted_state()
        coef = np.asarray(estimator.coef_, dtype=float).ravel()
        return pd.Series(coef, index=list(feature_names), name="coefficient")

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
        _, _, feature_names = self._require_fitted_state()
        return feature_names

    def _require_fitted_state(
        self,
    ) -> tuple[StandardScaler, BaseEstimator, tuple[str, ...]]:
        """Return fitted scaler/estimator/names or raise."""
        if (
            self._scaler is None
            or self._estimator is None
            or self._feature_names is None
        ):
            raise DataValidationError("ScaledLinearModel must be fitted before use.")
        return self._scaler, self._estimator, self._feature_names


class RidgeModel(ScaledLinearModel):
    """Ridge regression with train-only feature scaling.

    Parameters
    ----------
    alpha : float, default 1.0
        Ridge penalty strength.
    prediction_floor : float, default 1e-8
        Lower bound applied to predictions.

    Methods
    -------
    fit(features, target)
        Fit scaler and Ridge on training data.
    predict(features)
        Predict with the fitted Ridge model.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_RIDGE_ALPHA,
        prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    ) -> None:
        """Initialize an unfitted Ridge model.

        Parameters
        ----------
        alpha : float, default 1.0
            Ridge penalty strength.
        prediction_floor : float, default 1e-8
            Minimum allowed prediction.

        Raises
        ------
        DataValidationError
            If ``alpha`` is negative.
        """
        if alpha < 0:
            raise DataValidationError("Ridge alpha must be non-negative.")
        super().__init__(
            estimator=Ridge(alpha=alpha),
            prediction_floor=prediction_floor,
        )


class LassoModel(ScaledLinearModel):
    """Lasso regression with train-only feature scaling.

    Parameters
    ----------
    alpha : float, default 0.001
        Lasso penalty strength.
    prediction_floor : float, default 1e-8
        Lower bound applied to predictions.

    Methods
    -------
    fit(features, target)
        Fit scaler and Lasso on training data.
    predict(features)
        Predict with the fitted Lasso model.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_LASSO_ALPHA,
        prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    ) -> None:
        """Initialize an unfitted Lasso model.

        Parameters
        ----------
        alpha : float, default 0.001
            Lasso penalty strength.
        prediction_floor : float, default 1e-8
            Minimum allowed prediction.

        Raises
        ------
        DataValidationError
            If ``alpha`` is negative.
        """
        if alpha < 0:
            raise DataValidationError("Lasso alpha must be non-negative.")
        super().__init__(
            estimator=Lasso(alpha=alpha, max_iter=DEFAULT_MAX_ITER),
            prediction_floor=prediction_floor,
        )


class ElasticNetModel(ScaledLinearModel):
    """Elastic-net regression with train-only feature scaling.

    Parameters
    ----------
    alpha : float, default 0.001
        Penalty strength.
    l1_ratio : float, default 0.5
        Mixing parameter in ``[0, 1]`` (1 = Lasso-like).
    prediction_floor : float, default 1e-8
        Lower bound applied to predictions.

    Methods
    -------
    fit(features, target)
        Fit scaler and ElasticNet on training data.
    predict(features)
        Predict with the fitted ElasticNet model.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ELASTICNET_ALPHA,
        l1_ratio: float = DEFAULT_ELASTICNET_L1_RATIO,
        prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    ) -> None:
        """Initialize an unfitted ElasticNet model.

        Parameters
        ----------
        alpha : float, default 0.001
            Penalty strength.
        l1_ratio : float, default 0.5
            Elastic-net mixing parameter.
        prediction_floor : float, default 1e-8
            Minimum allowed prediction.

        Raises
        ------
        DataValidationError
            If ``alpha`` is negative or ``l1_ratio`` is outside ``[0, 1]``.
        """
        if alpha < 0:
            raise DataValidationError("ElasticNet alpha must be non-negative.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise DataValidationError("ElasticNet l1_ratio must be in [0, 1].")
        super().__init__(
            estimator=ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=DEFAULT_MAX_ITER,
            ),
            prediction_floor=prediction_floor,
        )
