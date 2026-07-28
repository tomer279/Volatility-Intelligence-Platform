"""Pydantic settings schema for VIP configuration.

Exports
-------
DateRangeConfig
    Inclusive request window from YAML.
PathsConfig
    Local data and artifact directories.
TargetConfig
    Realized-volatility target definition.
EvaluationConfig
    Primary and secondary metrics.
LoggingConfig
    Process logging settings.
AppConfig
    Top-level application configuration.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from vip.domain.enums import MetricName, RvEstimator


class DateRangeConfig(BaseModel):
    """Inclusive calendar window for data requests.

    Parameters
    ----------
    start : datetime.date
        First day of the sample.
    end : datetime.date or None, default None
        Last day of the sample. ``None`` means open-ended.
    """

    start: date
    end: date | None = None


class PathsConfig(BaseModel):
    """Filesystem locations for data and artifacts.

    Parameters
    ----------
    raw_dir : str
        Directory for raw vendor extracts.
    processed_dir : str
        Directory for normalized tables.
    artifacts_dir : str
        Directory for experiment outputs.
    """

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "data/artifacts"


class TargetConfig(BaseModel):
    """Realized-volatility prediction target.

    Parameters
    ----------
    horizon_days : int
        Forecast horizon in trading days.
    rv_estimator : RvEstimator
        Estimator used to build the target series.
    """

    horizon_days: int = Field(default=5, ge=1)
    rv_estimator: RvEstimator = RvEstimator.CLOSE_TO_CLOSE


class EvaluationConfig(BaseModel):
    """Metrics used to score forecasts.

    Parameters
    ----------
    primary_metric : MetricName
        Metric used for ranking and selection.
    secondary_metrics : list of MetricName
        Additional metrics reported alongside the primary.
    """

    primary_metric: MetricName = MetricName.QLIKE
    secondary_metrics: list[MetricName] = Field(
        default_factory=lambda: [MetricName.MSE, MetricName.MAE]
    )


class LoggingConfig(BaseModel):
    """Logging settings for CLI and pipelines.

    Parameters
    ----------
    level : str
        Stdlib logging level name (for example ``'INFO'``).
    """

    level: str = "INFO"


class AppConfig(BaseModel):
    """Top-level VIP application configuration.

    Parameters
    ----------
    symbol : str
        Primary study ticker.
    date_range : DateRangeConfig
        Sample window.
    paths : PathsConfig
        Data directories.
    target : TargetConfig
        Prediction target definition.
    evaluation : EvaluationConfig
        Scoring metrics.
    logging : LoggingConfig
        Logging behavior.
    """

    symbol: str = "SPY"
    date_range: DateRangeConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)