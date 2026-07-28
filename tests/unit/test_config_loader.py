"""Tests for YAML config loading."""

from vip.config import load_config
from vip.domain.enums import MetricName, RvEstimator


def test_load_default_config() -> None:
    """Default YAML loads with locked Milestone 0 research settings."""
    config = load_config()
    assert config.symbol == "SPY"
    assert config.target.horizon_days == 5
    assert config.target.rv_estimator == RvEstimator.CLOSE_TO_CLOSE
    assert config.evaluation.primary_metric == MetricName.QLIKE
    assert config.evaluation.secondary_metrics == [
        MetricName.MSE,
        MetricName.MAE,
    ]
    assert config.date_range.end is None
    assert config.logging.level == "INFO"