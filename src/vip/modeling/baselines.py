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
OuRvModel
    Frozen-origin discrete OU / AR(1) on log training target.
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
DEFAULT_OU_HORIZON_DAYS = 5
DEFAULT_OU_MIN_OBS = 30
MIN_OU_OBSERVATIONS = 3
PHI_CLIP_LIMIT = 0.999
LAG_SHIFT = 1
FIRST_OBSERVATION = 0
LAST_OBSERVATION = -1
LAG_COLUMN = "x_lag"
CONST_COLUMN = "const"
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


class OuRvModel:
    """Frozen-origin discrete OU / AR(1) baseline on log realized vol.

    State is ``x_t = log(y_t)`` on strictly positive training targets.
    Parameters ``(θ, φ)`` come from intercept OLS of ``x_t`` on ``x_{t-1}``.
    ``predict`` emits the same h-step mean for every row (frozen ``x_T``).

    Parameters
    ----------
    horizon_days : int, default 5
        Forecast horizon ``h`` used in the analytic h-step mean.
    prediction_floor : float, default 1e-8
        Lower bound applied after ``exp`` of the log-mean.
    min_obs : int, default 30
        Minimum finite positive training observations (need ≥ 3 for AR(1)
        with intercept).

    Methods
    -------
    fit(features, target)
        Estimate ``(θ, φ)`` on train-only log target and freeze ``x_T``.
    predict(features)
        Return the constant h-step forecast aligned to ``features.index``.
    fitted_state()
        Return ``(theta, phi, end_log_state)``.
    horizon_days()
        Return the forecast horizon ``h`` used in ``predict``.
    """

    def __init__(
            self,
            horizon_days: int = DEFAULT_OU_HORIZON_DAYS,
            prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
            min_obs: int = DEFAULT_OU_MIN_OBS,
    ) -> None:
        """Initialize an unfitted discrete OU model.

        Parameters
        ----------
        horizon_days : int, default 5
            Forecast horizon ``h``.
        prediction_floor : float, default 1e-8
            Minimum allowed prediction.
        min_obs : int, default 30
            Minimum finite positive training observations.

        Raises
        ------
        DataValidationError
            If ``horizon_days`` < 1, ``prediction_floor`` is not positive,
            or ``min_obs`` is below ``MIN_OU_OBSERVATIONS``.
        """
        if horizon_days < 1:
            raise DataValidationError("horizon_days must be at least 1.")
        if prediction_floor <= 0:
            raise DataValidationError("prediction_floor must be positive.")
        if min_obs < MIN_OU_OBSERVATIONS:
            raise DataValidationError(
                f"min_obs must be at least {MIN_OU_OBSERVATIONS}."
            )
        self._horizon_days = int(horizon_days)
        self._prediction_floor = float(prediction_floor)
        self._min_obs = int(min_obs)
        self._theta: float | None = None
        self._phi: float | None = None
        self._end_log_state: float | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> OuRvModel:
        """Fit discrete OU parameters on training log-target only.

        Feature columns are ignored. ``target`` is reindexed to
        ``features.index`` so values off the train index are unused.

        Parameters
        ----------
        features : pandas.DataFrame
            Training rows; only the index is used.
        target : pandas.Series
            Training realized-volatility target.

        Returns
        -------
        OuRvModel
            Fitted model (``self``).

        Raises
        ------
        DataValidationError
            If the aligned target is empty, non-positive, or too short.
        """
        aligned = target.reindex(features.index)
        log_state = _positive_log_state(aligned, self._min_obs)
        theta, phi = _estimate_ou_ar1(log_state)
        end_values = log_state.to_numpy(dtype=float, copy=True)
        self._theta = theta
        self._phi = phi
        self._end_log_state = float(end_values[LAST_OBSERVATION])
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict with the frozen h-step mean from end-of-train state.

        Parameters
        ----------
        features : pandas.DataFrame
            Feature rows to score (index alignment only).

        Returns
        -------
        pandas.Series
            Constant predictions aligned to ``features.index``, floored at
            ``prediction_floor``.

        Raises
        ------
        DataValidationError
            If the model has not been fitted.
        """
        theta, phi, origin_log_state = self._require_fitted()
        log_mean = _h_step_log_mean(
            theta,
            phi,
            origin_log_state,
            self._horizon_days,
        )
        yhat = max(float(np.exp(log_mean)), self._prediction_floor)
        return pd.Series(yhat, index=features.index, name="prediction")

    def fitted_state(self) -> tuple[float, float, float]:
        """Return fitted ``(theta, phi, end_log_state)``.

        Returns
        -------
        theta : float
            Estimated long-run mean of log-state.
        phi : float
            Clipped AR(1) coefficient.
        end_log_state : float
            Last finite training log-target ``x_T``.

        Raises
        ------
        DataValidationError
            If the model has not been fitted.
        """
        return self._require_fitted()

    def horizon_days(self) -> int:
        """Return the forecast horizon ``h`` used in ``predict``.
        Returns
        -------
        int
            Horizon in trading days.
        """
        return self._horizon_days

    def _require_fitted(self) -> tuple[float, float, float]:
        """Return fitted OU state or raise if unfitted."""
        if self._theta is None or self._phi is None or self._end_log_state is None:
            raise DataValidationError("OuRvModel must be fitted before predict.")
        return self._theta, self._phi, self._end_log_state


def _positive_log_state(target: pd.Series, min_obs: int) -> pd.Series:
    """Return log-state from strictly positive finite target values.

    Parameters
    ----------
    target : pandas.Series
        Training target already aligned to the feature index.
    min_obs : int
        Minimum number of finite positive observations.

    Returns
    -------
    pandas.Series
        ``log(y)`` on the finite positive subsample, in time order.

    Raises
    ------
    DataValidationError
        If the series is empty, or shorter than ``min_obs``.
    """
    values = target.to_numpy(dtype=float, copy=True)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        raise DataValidationError("Training target must contain finite values.")
    positive_mask = finite_values > 0.0
    positive_values = finite_values[positive_mask]
    if positive_values.size == 0:
        raise DataValidationError(
            "Training target must be strictly positive to form log-state."
        )
    if positive_values.size < min_obs:
        raise DataValidationError(
            f"OuRvModel requires at least {min_obs} finite positive training observations."
        )
    log_values = np.log(positive_values)
    positive_index = target.index[finite_mask][positive_mask]
    return pd.Series(
        log_values,
        index=positive_index,
        name="log_target",
    )


def _estimate_ou_ar1(log_state: pd.Series) -> tuple[float, float]:
    """Estimate discrete OU parameters via intercept OLS on lagged log-state.

    Fits ``x_t = α + φ x_{t-1} + ε`` and maps to
    ``θ = α / (1 - φ)`` with ``φ`` clipped to
    ``[-PHI_CLIP_LIMIT, PHI_CLIP_LIMIT]`` so the h-step map stays stable.

    Parameters
    ----------
    log_state : pandas.Series
        Finite log training target in time order.

    Returns
    -------
    theta : float
        Estimated long-run mean of log-state.
    phi : float
        Clipped AR(1) coefficient.

    Raises
    ------
    DataValidationError
        If a lagged pair cannot be formed.
    """
    values = log_state.to_numpy(dtype=float, copy=True)
    x_now = values[LAG_SHIFT:]
    x_lag = values[:-LAG_SHIFT]
    if x_now.size == 0:
        raise DataValidationError(
            "OuRvModel requires a lagged pair to fit discrete OU / AR(1)."
        )
    lag_frame = pd.DataFrame({LAG_COLUMN: x_lag})
    design = sm.add_constant(lag_frame, has_constant="add")
    result = sm.OLS(x_now, design).fit()
    intercept = float(result.params[CONST_COLUMN])
    phi_raw = float(result.params[LAG_COLUMN])
    phi = float(np.clip(phi_raw, -PHI_CLIP_LIMIT, PHI_CLIP_LIMIT))
    theta = intercept / (1.0 - phi)
    return theta, phi


def _h_step_log_mean(
        theta: float,
        phi: float,
        origin_log_state: float,
        horizon_days: int,
) -> float:
    """Return the analytic h-step conditional mean on log-state.

    Parameters
    ----------
    theta : float
        Long-run mean of log-state.
    phi : float
        AR(1) coefficient.
    origin_log_state : float
        Frozen origin state ``x_T``.
    horizon_days : int
        Forecast horizon ``h``.

    Returns
    -------
    float
        ``θ + φ^h (x_T − θ)``.
    """
    reversion = phi ** horizon_days
    return theta + reversion * (origin_log_state - theta)
