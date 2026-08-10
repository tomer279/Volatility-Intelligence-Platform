"""Baseline volatility forecasting models.

Exports
-------
HistoricalMeanModel
    Constant forecast equal to the training-target mean.
EwmaModel
    Frozen end-of-train EWMA level forecast.
HarRvOlsModel
    OLS HAR-RV model on trailing RV feature columns.
VixAsForecastModel
    Intercept OLS of target on daily VIX vol (``vix_as_forecast``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from vip.domain.errors import DataValidationError
from vip.features.cross_asset import VIX_LEVEL_COLUMN
from vip.features.iv_rv_features import (
    VIX_VOL_DAILY_COLUMN,
    vix_level_to_daily_vol,
)

DEFAULT_EWMA_LAMBDA = 0.94
DEFAULT_PREDICTION_FLOOR = 1e-8
HAR_FEATURE_COLUMNS: tuple[str, ...] = ("rv_cc_1d", "rv_cc_5d", "rv_cc_21d")
VIX_AS_FORECAST_REGRESSOR = VIX_VOL_DAILY_COLUMN


class HistoricalMeanModel:
    """Constant forecast using the mean of training targets.

    Methods
    -------
    fit(_features, target)
        Store the training-target mean.
    predict(features)
        Return a constant series aligned to ``features.index``.
    """

    def __init__(self) -> None:
        """Initialize an unfitted historical-mean model."""
        self._mean: float | None = None

    def fit(self, _features: pd.DataFrame, target: pd.Series) -> HistoricalMeanModel:
        """Fit the model on training data.

        Parameters
        ----------
        _features : pandas.DataFrame
            Training features (ignored by this baseline).
        target : pandas.Series
            Training realized-volatility target.

        Returns
        -------
        HistoricalMeanModel
            Fitted model (``self``).

        Raises
        ------
        DataValidationError
            If ``target`` has no finite values.
        """
        clean_target = target.dropna()
        if clean_target.empty:
            raise DataValidationError("Training target must contain finite values.")
        self._mean = float(clean_target.mean())
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict using the fitted training mean.

        Parameters
        ----------
        features : pandas.DataFrame
            Feature rows to score.

        Returns
        -------
        pandas.Series
            Constant predictions aligned to ``features.index``.

        Raises
        ------
        DataValidationError
            If the model has not been fitted.
        """
        mean_value = self._require_mean()
        return pd.Series(mean_value, index=features.index, name="prediction")

    def _require_mean(self) -> float:
        """Return the fitted mean or raise if unfitted."""
        if self._mean is None:
            raise DataValidationError("HistoricalMeanModel must be fitted before predict.")
        return self._mean


class EwmaModel:
    """EWMA baseline with forecast frozen at the end of training.

    Parameters
    ----------
    decay : float, default 0.94
        EWMA decay factor ``lambda`` in ``(0, 1)``.

    Methods
    -------
    fit(_features, target)
        Compute the end-of-train EWMA level from ``target``.
    predict(features)
        Return that frozen level for all requested rows.
    """

    def __init__(self, decay: float = DEFAULT_EWMA_LAMBDA) -> None:
        """Initialize an EWMA model.

        Parameters
        ----------
        decay : float, default 0.94
            EWMA decay factor.

        Raises
        ------
        DataValidationError
            If ``decay`` is not in ``(0, 1)``.
        """
        if not 0.0 < decay < 1.0:
            raise DataValidationError("EWMA decay must be in the open interval (0, 1).")
        self._decay = decay
        self._level: float | None = None

    def fit(self, _features: pd.DataFrame, target: pd.Series) -> EwmaModel:
        """Fit the EWMA level on training targets.

        Parameters
        ----------
        _features : pandas.DataFrame
            Training features (ignored by this baseline).
        target : pandas.Series
            Training realized-volatility target.

        Returns
        -------
        EwmaModel
            Fitted model (``self``).

        Raises
        ------
        DataValidationError
            If ``target`` has no finite values.
        """
        clean_target = target.dropna()
        if clean_target.empty:
            raise DataValidationError("Training target must contain finite values.")

        values = clean_target.to_numpy(dtype=float)
        level = float(values[0])
        for value in values[1:]:
            level = self._decay * level + (1.0 - self._decay) * float(value)
        self._level = level
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict using the frozen end-of-train EWMA level.

        Parameters
        ----------
        features : pandas.DataFrame
            Feature rows to score.

        Returns
        -------
        pandas.Series
            Constant predictions aligned to ``features.index``.

        Raises
        ------
        DataValidationError
            If the model has not been fitted.
        """
        level = self._require_level()
        return pd.Series(level, index=features.index, name="prediction")

    def _require_level(self) -> float:
        """Return the fitted EWMA level or raise if unfitted."""
        if self._level is None:
            raise DataValidationError("EwmaModel must be fitted before predict.")
        return self._level


class HarRvOlsModel:
    """HAR-RV OLS model using trailing RV feature columns.

    Parameters
    ----------
    feature_columns : tuple of str, optional
        Regressor columns. Defaults to ``rv_cc_1d``, ``rv_cc_5d``, ``rv_cc_21d``.
    prediction_floor : float, default 1e-8
        Lower bound applied to predictions.

    Methods
    -------
    fit(features, target)
        Fit an intercept OLS model on HAR columns.
    predict(features)
        Predict with the fitted coefficients.
    """

    def __init__(
        self,
        feature_columns: tuple[str, ...] = HAR_FEATURE_COLUMNS,
        prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    ) -> None:
        """Initialize an unfitted HAR-RV OLS model.

        Parameters
        ----------
        feature_columns : tuple of str, optional
            Regressor column names.
        prediction_floor : float, default 1e-8
            Minimum allowed prediction.
        """
        if prediction_floor <= 0:
            raise DataValidationError("prediction_floor must be positive.")
        if not feature_columns:
            raise DataValidationError("feature_columns must be non-empty.")
        self._feature_columns = feature_columns
        self._prediction_floor = prediction_floor
        self._params: pd.Series | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> HarRvOlsModel:
        """Fit OLS of the target on HAR feature columns plus intercept.

        Parameters
        ----------
        features : pandas.DataFrame
            Training features containing HAR columns.
        target : pandas.Series
            Training realized-volatility target.

        Returns
        -------
        HarRvOlsModel
            Fitted model (``self``).

        Raises
        ------
        DataValidationError
            If required columns are missing or training data is empty.
        """
        design, clean_target = self._design_matrix(features, target)
        model = sm.OLS(clean_target, design)
        result = model.fit()
        self._params = result.params
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict realized volatility with the fitted OLS model.

        Parameters
        ----------
        features : pandas.DataFrame
            Feature rows containing HAR columns.

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
        params = self._require_params()
        missing = [column for column in self._feature_columns if column not in features.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise DataValidationError(
                f"Missing HAR feature columns for prediction: {missing_text}."
            )

        design = sm.add_constant(features.loc[:, list(self._feature_columns)], has_constant="add")
        # Ensure column order matches training params.
        design = design.loc[:, params.index]
        raw = design.to_numpy(dtype=float) @ params.to_numpy(dtype=float)
        clipped = np.maximum(raw, self._prediction_floor)
        return pd.Series(clipped, index=features.index, name="prediction")

    def _design_matrix(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build an aligned OLS design matrix and target vector."""
        missing = [column for column in self._feature_columns if column not in features.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise DataValidationError(
                f"Missing HAR feature columns for fitting: {missing_text}."
            )

        frame = features.loc[:, list(self._feature_columns)].copy()
        frame["target"] = target
        clean = frame.dropna()
        if clean.empty:
            raise DataValidationError("No finite rows available to fit HarRvOlsModel.")

        clean_target = clean["target"]
        design = sm.add_constant(clean.loc[:, list(self._feature_columns)], has_constant="add")
        return design, clean_target

    def _require_params(self) -> pd.Series:
        """Return fitted parameters or raise if unfitted."""
        if self._params is None:
            raise DataValidationError("HarRvOlsModel must be fitted before predict.")
        return self._params


class VixAsForecastModel:
    """Implied-as-forecast baseline: intercept OLS on daily VIX vol.

    Uses ``vix_vol_daily`` when present; otherwise derives it from
    ``vix_level`` via ``vix_level_to_daily_vol``. Same fit/predict surface
    as other horse-race models.

    Parameters
    ----------
    prediction_floor : float, default 1e-8
        Lower bound applied to predictions.

    Methods
    -------
    fit(features, target)
        Fit intercept OLS of ``target`` on daily VIX vol.
    predict(features)
        Predict with fitted coefficients, floored at ``prediction_floor``.
    """

    def __init__(
        self,
        prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    ) -> None:
        """Initialize an unfitted VIX-as-forecast model.

        Parameters
        ----------
        prediction_floor : float, default 1e-8
            Minimum allowed prediction.

        Raises
        ------
        DataValidationError
            If ``prediction_floor`` is not positive.
        """
        if prediction_floor <= 0:
            raise DataValidationError("prediction_floor must be positive.")
        self._prediction_floor = prediction_floor
        self._params: pd.Series | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> VixAsForecastModel:
        """Fit OLS of the target on daily VIX vol plus intercept.

        Parameters
        ----------
        features : pandas.DataFrame
            Training features with ``vix_vol_daily`` and/or ``vix_level``.
        target : pandas.Series
            Training realized-volatility target.

        Returns
        -------
        VixAsForecastModel
            Fitted model (``self``).

        Raises
        ------
        DataValidationError
            If the predictor cannot be resolved or training rows are empty.
        """
        design, clean_target = self._design_matrix(features, target)
        result = sm.OLS(clean_target, design).fit()
        self._params = result.params
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict realized volatility from daily VIX vol.

        Parameters
        ----------
        features : pandas.DataFrame
            Feature rows with ``vix_vol_daily`` and/or ``vix_level``.

        Returns
        -------
        pandas.Series
            Predictions aligned to ``features.index``, floored at
            ``prediction_floor``.

        Raises
        ------
        DataValidationError
            If the model is unfitted or the predictor cannot be resolved.
        """
        params = self._require_params()
        predictor = self._resolve_predictor(features)
        design = sm.add_constant(
            predictor.to_frame(name=VIX_AS_FORECAST_REGRESSOR),
            has_constant="add",
        )
        design = design.loc[:, params.index]
        raw = design.to_numpy(dtype=float) @ params.to_numpy(dtype=float)
        clipped = np.maximum(raw, self._prediction_floor)
        return pd.Series(clipped, index=features.index, name="prediction")

    def _resolve_predictor(self, features: pd.DataFrame) -> pd.Series:
        """Return daily VIX vol from ``vix_vol_daily`` or ``vix_level``."""
        if VIX_VOL_DAILY_COLUMN in features.columns:
            return features[VIX_VOL_DAILY_COLUMN]
        if VIX_LEVEL_COLUMN in features.columns:
            return vix_level_to_daily_vol(features[VIX_LEVEL_COLUMN])
        raise DataValidationError(
            "VixAsForecastModel requires 'vix_vol_daily' or 'vix_level'."
        )

    def _design_matrix(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build an aligned OLS design matrix and target vector."""
        predictor = self._resolve_predictor(features)
        frame = pd.DataFrame(
            {
                VIX_AS_FORECAST_REGRESSOR: predictor,
                "target": target,
            },
            index=features.index,
        )
        clean = frame.dropna()
        if clean.empty:
            raise DataValidationError(
                "No finite rows available to fit VixAsForecastModel."
            )
        clean_target = clean["target"]
        design = sm.add_constant(
            clean.loc[:, [VIX_AS_FORECAST_REGRESSOR]],
            has_constant="add",
        )
        return design, clean_target

    def _require_params(self) -> pd.Series:
        """Return fitted parameters or raise if unfitted."""
        if self._params is None:
            raise DataValidationError(
                "VixAsForecastModel must be fitted before predict."
            )
        return self._params
