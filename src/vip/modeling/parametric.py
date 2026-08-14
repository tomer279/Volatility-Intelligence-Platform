"""Parametric / filter baselines for Milestone 10 stretch.

Exports
-------
EwmaRecursiveModel
    Train-fit EWMA decay; recursive OOS updates via trailing RV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vip.domain.errors import DataValidationError
from vip.modeling.baselines import (
    DEFAULT_PREDICTION_FLOOR,
    FIRST_OBSERVATION,
    LAG_SHIFT,
)

DEFAULT_RECURSIVE_OBS_COLUMN = "rv_cc_1d"
DEFAULT_RECURSIVE_MIN_OBS = 30
MIN_RECURSIVE_OBSERVATIONS = 2
DEFAULT_DECAY_GRID: tuple[float, ...] = (
    0.80,
    0.85,
    0.90,
    0.94,
    0.97,
    0.99,
)


class EwmaRecursiveModel:
    """Train-fit EWMA decay with recursive OOS state updates.

    Distinct from ``EwmaModel`` (fixed ``lambda``, frozen level for all
    test rows). Decay is chosen on train targets only. ``predict`` updates
    state from a trailing RV feature column (default ``rv_cc_1d``), never
    from the forward label.

    Parameters
    ----------
    observation_column : str, default ``rv_cc_1d``
        Trailing RV feature used to update state during ``predict``.
    prediction_floor : float, default 1e-8
        Lower bound applied to predictions.
    min_obs : int, default 30
        Minimum finite training observations.

    Methods
    -------
    fit(_features, target)
        Estimate decay on train targets; freeze end-of-train EWMA level.
    predict(features)
        Recurse state with ``observation_column``; return floored forecasts.
    fitted_decay()
        Return the train-selected decay.
    fitted_level()
        Return the end-of-train EWMA level before OOS updates.
    """

    def __init__(
            self,
            observation_column: str = DEFAULT_RECURSIVE_OBS_COLUMN,
            prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
            min_obs: int = DEFAULT_RECURSIVE_MIN_OBS,
    ) -> None:
        """Initialize an unfitted recursive EWMA filter.

        Parameters
        ----------
        observation_column : str, default ``rv_cc_1d``
            Trailing RV column for OOS state updates.
        prediction_floor : float, default 1e-8
            Minimum allowed prediction.
        min_obs : int, default 30
            Minimum finite training observations.

        Raises
        ------
        DataValidationError
            If ``observation_column`` is empty, ``prediction_floor`` is not
            positive, or ``min_obs`` is below ``MIN_RECURSIVE_OBSERVATIONS``.
        """
        if not observation_column:
            raise DataValidationError("observation_column must be non-empty.")
        if prediction_floor <= 0:
            raise DataValidationError("prediction_floor must be positive.")
        if min_obs < MIN_RECURSIVE_OBSERVATIONS:
            raise DataValidationError(
                "min_obs must be at least "
                f"{MIN_RECURSIVE_OBSERVATIONS}."
            )
        self._observation_column = str(observation_column)
        self._prediction_floor = float(prediction_floor)
        self._min_obs = int(min_obs)
        self._decay: float | None = None
        self._end_level: float | None = None

    def fit(
            self,
            _features: pd.DataFrame,
            target: pd.Series,
    ) -> EwmaRecursiveModel:
        """Fit decay and end-of-train level on training targets only.

        Feature columns are ignored for estimation. ``target`` is reindexed
        to ``_features.index`` so values off the train index are unused.

        Parameters
        ----------
        _features : pandas.DataFrame
            Training rows; only the index is used for alignment.
        target : pandas.Series
            Training realized-volatility target.

        Returns
        -------
        EwmaRecursiveModel
            Fitted model (``self``).

        Raises
        ------
        DataValidationError
            If the aligned target is empty or shorter than ``min_obs``.
        """
        aligned = target.reindex(_features.index)
        values = _finite_train_values(aligned, self._min_obs)
        decay = _select_decay(values, DEFAULT_DECAY_GRID)
        self._decay = decay
        self._end_level = _ewma_terminal_level(values, decay)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict with recursive filter updates on trailing RV.

        For each row in index order: update state with a finite observation
        from ``observation_column`` (if present), then emit the floored
        level. Does not read target / label columns.

        Parameters
        ----------
        features : pandas.DataFrame
            Feature rows containing ``observation_column``.

        Returns
        -------
        pandas.Series
            Predictions aligned to ``features.index``.

        Raises
        ------
        DataValidationError
            If unfitted or ``observation_column`` is missing.
        """
        decay, level = self._require_fitted()
        if self._observation_column not in features.columns:
            raise DataValidationError(
                "Missing observation column for EwmaRecursiveModel: "
                f"{self._observation_column}."
            )
        observations = features[self._observation_column].to_numpy(
            dtype=float,
            copy=True,
        )
        predictions = np.empty(observations.shape[0], dtype=float)
        state = float(level)
        for _row_index, observation in enumerate(observations):
            if np.isfinite(observation):
                state = decay * state + (1.0 - decay) * float(observation)
            predictions[_row_index] = max(state, self._prediction_floor)
        return pd.Series(predictions, index=features.index, name="prediction")

    def fitted_decay(self) -> float:
        """Return the train-selected EWMA decay.

        Returns
        -------
        float
            Fitted decay in ``(0, 1)``.

        Raises
        ------
        DataValidationError
            If the model has not been fitted.
        """
        decay, _level = self._require_fitted()
        return decay

    def fitted_level(self) -> float:
        """Return the end-of-train EWMA level (pre-OOS updates).

        Returns
        -------
        float
            Frozen train-terminal filter level.

        Raises
        ------
        DataValidationError
            If the model has not been fitted.
        """
        _decay, level = self._require_fitted()
        return level

    def _require_fitted(self) -> tuple[float, float]:
        """Return ``(decay, end_level)`` or raise if unfitted."""
        if self._decay is None or self._end_level is None:
            raise DataValidationError(
                "EwmaRecursiveModel must be fitted before predict."
            )
        return self._decay, self._end_level


def _finite_train_values(target: pd.Series, min_obs: int) -> np.ndarray:
    """Return finite training target values in time order.

    Parameters
    ----------
    target : pandas.Series
        Training target already aligned to the feature index.
    min_obs : int
        Minimum number of finite observations.

    Returns
    -------
    numpy.ndarray
        Finite float values.

    Raises
    ------
    DataValidationError
        If empty or shorter than ``min_obs``.
    """
    values = target.to_numpy(dtype=float, copy=True)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise DataValidationError("Training target must contain finite values.")
    if finite_values.size < min_obs:
        raise DataValidationError(
            "EwmaRecursiveModel requires at least "
            f"{min_obs} finite training observations."
        )
    return finite_values


def _ewma_terminal_level(values: np.ndarray, decay: float) -> float:
    """Return the EWMA level after consuming ``values`` left to right."""
    level = float(values[FIRST_OBSERVATION])
    for value in values[LAG_SHIFT:]:
        level = decay * level + (1.0 - decay) * float(value)
    return float(level)


def _one_step_ewma_mse(values: np.ndarray, decay: float) -> float:
    """Return mean squared one-step EWMA forecast error on ``values``."""
    level = float(values[FIRST_OBSERVATION])
    squared_errors: list[float] = []
    for value in values[LAG_SHIFT:]:
        residual = level - float(value)
        squared_errors.append(residual * residual)
        level = decay * level + (1.0 - decay) * float(value)
    return float(np.mean(np.asarray(squared_errors, dtype=float)))


def _select_decay(values: np.ndarray, grid: tuple[float, ...]) -> float:
    """Return the grid decay with lowest one-step EWMA MSE."""
    best_decay = float(grid[FIRST_OBSERVATION])
    best_mse = _one_step_ewma_mse(values, best_decay)
    for decay in grid[LAG_SHIFT:]:
        mse = _one_step_ewma_mse(values, float(decay))
        if mse < best_mse:
            best_mse = mse
            best_decay = float(decay)
    return best_decay
