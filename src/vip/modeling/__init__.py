"""Volatility forecasting models and baselines.

Exports
-------
HistoricalMeanModel
    Constant training-mean forecast.
EwmaModel
    Frozen end-of-train EWMA forecast.
HarRvOlsModel
    HAR-RV OLS baseline.
"""

from vip.modeling.baselines import EwmaModel, HarRvOlsModel, HistoricalMeanModel

__all__ = [
    "EwmaModel",
    "HarRvOlsModel",
    "HistoricalMeanModel",
]