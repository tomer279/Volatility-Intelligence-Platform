"""Volatility forecasting models and baselines.

Exports
-------
HistoricalMeanModel
    Constant training-mean forecast.
EwmaModel
    Frozen end-of-train EWMA forecast.
HarRvOlsModel
    HAR-RV OLS baseline.
ScaledLinearModel
    Scaled scikit-learn linear adapter.
RidgeModel
    Ridge regression adapter.
LassoModel
    Lasso regression adapter.
ElasticNetModel
    Elastic-net regression adapter.
ModelSpec
    Metadata and factory for one registered model.
ModelRegistry
    Register factories and create model instances.
create_default_model_registry
    Build the default model registry.
"""

from vip.modeling.baselines import EwmaModel, HarRvOlsModel, HistoricalMeanModel
from vip.modeling.registry import (
    ModelRegistry,
    ModelSpec,
    create_default_model_registry,
)
from vip.modeling.regularization import (
    ElasticNetModel,
    LassoModel,
    RidgeModel,
    ScaledLinearModel,
)

__all__ = [
    "ElasticNetModel",
    "EwmaModel",
    "HarRvOlsModel",
    "HistoricalMeanModel",
    "LassoModel",
    "ModelRegistry",
    "ModelSpec",
    "RidgeModel",
    "ScaledLinearModel",
    "create_default_model_registry",
]
