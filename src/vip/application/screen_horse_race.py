"""Horse-race walk-forward and M7 inference for factor screens.

Exports
-------
HORSE_RACE_MODELS
    Locked catalog of competing forecast models (incl. ``ou_rv``,
    ``ewma_recursive``, and ``vix_as_forecast``).
VIX_AS_FORECAST_MODEL
    Registry name for the VIX-as-forecast baseline.
OU_RV_MODEL
    Registry name for the discrete OU / AR(1) baseline.
EWMA_RECURSIVE_MODEL
    Registry name for the stretch recursive EWMA filter.
HorseRaceOptions
    Walk-forward split settings plus inference summary options.
run_horse_race_with_inference
    Fit/race models, attach OOS QLIKE losses, and enrich the summary.
resolve_horse_race_models
    Build the race dict; omit ``vix_as_forecast`` without VIX predictors.
    ``ou_rv`` and ``ewma_recursive`` are always included; ``horizon_days``
    is injected into ``ou_rv``.
features_support_vix_as_forecast
    Return whether the feature matrix has ``vix_vol_daily`` or ``vix_level``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.evaluation.comparison import (
    InferenceSummaryOptions,
    summarize_with_inference,
)
from vip.evaluation.walk_forward import (
    attach_qlike_losses,
    collect_walk_forward_predictions,
    run_walk_forward,
)
from vip.features.cross_asset import VIX_LEVEL_COLUMN
from vip.features.iv_rv_features import VIX_VOL_DAILY_COLUMN
from vip.modeling.registry import create_default_model_registry
from vip.modeling.baselines import DEFAULT_OU_HORIZON_DAYS, OuRvModel


HORSE_RACE_MODELS = (
    "har_rv_ols", "ridge", "lasso", "vix_as_forecast", "ou_rv", "ewma_recursive",
)
VIX_AS_FORECAST_MODEL = "vix_as_forecast"
OU_RV_MODEL = "ou_rv"
EWMA_RECURSIVE_MODEL = "ewma_recursive"


@dataclass(frozen=True, slots=True)
class HorseRaceOptions:
    """Walk-forward and inference settings for one horse-race run.

    Parameters
    ----------
    n_splits : int
        Number of expanding walk-forward folds.
    embargo_size : int
        Embargo length in sessions between train and test.
    summary_options : InferenceSummaryOptions
        Options consumed by ``summarize_with_inference``.

    Methods
    -------
    describe()
        Return a short human-readable summary.
    validate()
        Raise if split settings are invalid.
    """

    n_splits: int
    embargo_size: int
    summary_options: InferenceSummaryOptions

    def describe(self) -> str:
        """Return a short human-readable summary."""
        return (
            f"n_splits={self.n_splits}, embargo={self.embargo_size}, "
            f"{self.summary_options.describe()}"
        )

    def validate(self) -> None:
        """Raise ``DataValidationError`` when split settings are invalid."""
        if self.n_splits < 2:
            raise DataValidationError("n_splits must be at least 2.")
        if self.embargo_size < 0:
            raise DataValidationError("embargo_size must be non-negative.")
        self.summary_options.validate()


def run_horse_race_with_inference(
        features: pd.DataFrame,
        target: pd.Series,
        options: HorseRaceOptions,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walk-forward metrics, OOS losses, and inference-enriched summary.

    Parameters
    ----------
    features : pandas.DataFrame
        Screen feature matrix.
    target : pandas.Series
        Target aligned to ``features``.
    options : HorseRaceOptions
        Fold counts, embargo, and inference summary options.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(fold_metrics, oos_losses, summary)``.
    """
    options.validate()
    models = resolve_horse_race_models(
        features,
        horizon_days=options.summary_options.horizon_days,
    )
    fold_metrics = run_walk_forward(
        features=features,
        target=target,
        models=models,
        n_splits=options.n_splits,
        embargo_size=options.embargo_size,
    )
    predictions = collect_walk_forward_predictions(
        features=features,
        target=target,
        models=models,
        n_splits=options.n_splits,
        embargo_size=options.embargo_size,
    )
    oos_losses = attach_qlike_losses(predictions)
    summary = summarize_with_inference(
        fold_metrics,
        oos_losses,
        options=options.summary_options,
    )
    return fold_metrics, oos_losses, summary


def resolve_horse_race_models(
        features: pd.DataFrame,
        horizon_days: int = DEFAULT_OU_HORIZON_DAYS,
) -> dict[str, object]:
    """Create horse-race models, skipping VIX forecast without predictors.

    Parameters
    ----------
    features : pandas.DataFrame
        Screen feature matrix (may or may not include VIX columns).
    horizon_days : int, default 5
        Forecast horizon injected into ``ou_rv``. Other race models ignore
        this. The default registry factory also uses h=5.

    Returns
    -------
    dict of str to object
        Mapping of model name to instance from the default registry.
        ``vix_as_forecast`` is included only when ``vix_vol_daily`` or
        ``vix_level`` is present.
    """
    registry = create_default_model_registry()
    names = list(HORSE_RACE_MODELS)
    if not features_support_vix_as_forecast(features):
        names = [name for name in names if name != VIX_AS_FORECAST_MODEL]
    models = registry.create_many(names)
    if OU_RV_MODEL in models:
        models[OU_RV_MODEL] = OuRvModel(horizon_days=horizon_days)
    return models


def features_support_vix_as_forecast(features: pd.DataFrame) -> bool:
    """Return True when VixAsForecastModel can resolve a predictor column."""
    columns = set(features.columns)
    return (
        VIX_VOL_DAILY_COLUMN in columns
        or VIX_LEVEL_COLUMN in columns
    )
