"""Volatility forecasting models and baselines.

Exports
-------
HistoricalMeanModel
    Constant training-mean forecast.
EwmaModel
    Frozen end-of-train EWMA forecast.
HarRvOlsModel
    HAR-RV OLS baseline.
VixAsForecastModel
    Intercept OLS on daily VIX vol (registry name ``vix_as_forecast``).
ScaledLinearModel
    Scaled scikit-learn linear adapter.
RidgeModel
    Ridge regression adapter.
LassoModel
    Lasso regression adapter.
ElasticNetModel
    Elastic-net regression adapter.
TreeVolModel
    Unscaled tree regressor adapter.
RandomForestVolModel
    Random-forest volatility adapter.
ModelSpec
    Metadata and factory for one registered model.
ModelRegistry
    Register factories and create model instances.
create_default_model_registry
    Build the default model registry.
"""

from vip.modeling.baselines import (
    EwmaModel,
    HarRvOlsModel,
    HistoricalMeanModel,
    VixAsForecastModel,
)
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
from vip.modeling.tree_models import RandomForestVolModel, TreeVolModel

__all__ = [
    "ElasticNetModel",
    "EwmaModel",
    "HarRvOlsModel",
    "HistoricalMeanModel",
    "VixAsForecastModel",
    "LassoModel",
    "ModelRegistry",
    "ModelSpec",
    "RandomForestVolModel",
    "RidgeModel",
    "ScaledLinearModel",
    "TreeVolModel",
    "create_default_model_registry",
]
