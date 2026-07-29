"""Registry for named volatility model factories.

Exports
-------
ModelSpec
    Metadata and zero-arg factory for one model.
ModelRegistry
    Register factories and create fresh model instances.
create_default_model_registry
    Build a registry preloaded with baseline and regularized models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vip.domain.errors import DataValidationError
from vip.modeling.baselines import EwmaModel, HarRvOlsModel, HistoricalMeanModel
from vip.modeling.regularization import ElasticNetModel, LassoModel, RidgeModel
from vip.modeling.tree_models import RandomForestVolModel

ModelFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Specification for a registered volatility model.

    Parameters
    ----------
    name : str
        Registry key used in configs and experiment runners.
    factory : callable
        Zero-argument callable that returns a fresh unfitted model.
    description : str
        Short human-readable description.

    Methods
    -------
    create()
        Instantiate a new model from ``factory``.
    describe()
        Return the human-readable description.
    """

    name: str
    factory: ModelFactory
    description: str

    def create(self) -> Any:
        """Instantiate a new unfitted model.

        Returns
        -------
        Any
            Object exposing ``fit`` / ``predict``.
        """
        return self.factory()

    def describe(self) -> str:
        """Return the human-readable description.

        Returns
        -------
        str
            Model description text.
        """
        return self.description


class ModelRegistry:
    """In-memory registry of volatility model factories.

    Methods
    -------
    register(spec)
        Add or replace a model specification.
    get(name)
        Retrieve a registered specification.
    list_names()
        Return registered model names in insertion order.
    create(name)
        Build one fresh model instance by name.
    create_many(names)
        Build a name → instance mapping for walk-forward runners.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._specs: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        """Add or replace a model specification.

        Parameters
        ----------
        spec : ModelSpec
            Model to register.
        """
        self._specs[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        """Retrieve a registered specification.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        ModelSpec
            Registered specification.

        Raises
        ------
        DataValidationError
            If ``name`` is not registered.
        """
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(self.list_names()) or "<none>"
            raise DataValidationError(
                f"Unknown model '{name}'. Available: {available}."
            ) from exc

    def list_names(self) -> list[str]:
        """Return registered model names in insertion order.

        Returns
        -------
        list of str
            Registry keys.
        """
        return list(self._specs.keys())

    def create(self, name: str) -> Any:
        """Build one fresh model instance by name.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        Any
            Fresh unfitted model.
        """
        return self.get(name).create()

    def create_many(self, names: list[str]) -> dict[str, Any]:
        """Build a name → instance mapping.

        Parameters
        ----------
        names : list of str
            Registry keys to instantiate.

        Returns
        -------
        dict of str to Any
            Fresh models keyed by registry name.
        """
        return {name: self.create(name) for name in names}


def create_default_model_registry() -> ModelRegistry:
    """Create a registry preloaded with Milestone 3/4 models.

    Returns
    -------
    ModelRegistry
        Registry containing baselines and regularized linear models.
    """
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            name="historical_mean",
            factory=HistoricalMeanModel,
            description="Constant forecast equal to the training-target mean.",
        )
    )
    registry.register(
        ModelSpec(
            name="ewma",
            factory=EwmaModel,
            description="Frozen end-of-train EWMA level forecast.",
        )
    )
    registry.register(
        ModelSpec(
            name="har_rv_ols",
            factory=HarRvOlsModel,
            description="HAR-RV OLS on trailing RV feature columns.",
        )
    )
    registry.register(
        ModelSpec(
            name="ridge",
            factory=RidgeModel,
            description="Ridge regression with train-only StandardScaler.",
        )
    )
    registry.register(
        ModelSpec(
            name="lasso",
            factory=LassoModel,
            description="Lasso regression with train-only StandardScaler.",
        )
    )
    registry.register(
        ModelSpec(
            name="elasticnet",
            factory=ElasticNetModel,
            description="Elastic-net regression with train-only StandardScaler.",
        )
    )
    registry.register(
        ModelSpec(
            name="random_forest",
            factory=RandomForestVolModel,
            description="Random forest on unscaled features.",
        )
    )
    return registry
