"""Registry for named feature builders.

Exports
-------
FeatureSpec
    Metadata and builder callable for one feature family.
FeatureRegistry
    Register builders and assemble selected feature columns.
create_default_registry
    Build a registry with Milestone 2 families; optional ``jump`` via
    ``include_jump=True``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.features.har import build_har_features
from vip.features.jump_features import build_jump_features
from vip.features.range_features import build_range_features
from vip.features.returns import build_return_features
from vip.features.volume_features import build_volume_features

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Specification for a registered feature family.

    Parameters
    ----------
    name : str
        Registry key used in configs and CLI selections.
    builder : callable
        Function that maps OHLCV to a feature DataFrame.
    description : str
        Short human-readable description.

    Methods
    -------
    build(ohlcv)
        Run the builder on an OHLCV frame.
    """

    name: str
    builder: FeatureBuilder
    description: str

    def build(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Build features for this family.

        Parameters
        ----------
        ohlcv : pandas.DataFrame
            Canonical OHLCV frame.

        Returns
        -------
        pandas.DataFrame
            Feature columns produced by this family.
        """
        return self.builder(ohlcv)


class FeatureRegistry:
    """In-memory registry of feature-family builders.

    Methods
    -------
    register(spec)
        Add or replace a feature family specification.
    get(name)
        Retrieve a registered specification.
    list_names()
        Return registered family names in insertion order.
    build_all(ohlcv, names=None)
        Build and concatenate selected feature families.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        """Add or replace a feature family specification.

        Parameters
        ----------
        spec : FeatureSpec
            Feature family to register.
        """
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        """Retrieve a registered specification.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        FeatureSpec
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
                f"Unknown feature family '{name}'. Available: {available}."
            ) from exc

    def list_names(self) -> list[str]:
        """Return registered family names in insertion order.

        Returns
        -------
        list of str
            Registry keys.
        """
        return list(self._specs.keys())

    def build_all(
        self,
        ohlcv: pd.DataFrame,
        names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build and concatenate selected feature families.

        Parameters
        ----------
        ohlcv : pandas.DataFrame
            Canonical OHLCV frame.
        names : list of str or None, default None
            Family names to build. ``None`` builds all registered families.

        Returns
        -------
        pandas.DataFrame
            Concatenated feature columns aligned on the OHLCV index.

        Raises
        ------
        DataValidationError
            If a requested name is unknown or a builder returns no columns.
        """
        selected = names if names is not None else self.list_names()
        if not selected:
            raise DataValidationError("No feature families selected to build.")

        frames: list[pd.DataFrame] = []
        for name in selected:
            frame = self.get(name).build(ohlcv)
            if frame.empty or frame.shape[1] == 0:
                raise DataValidationError(
                    f"Feature family '{name}' returned no columns."
                )
            frames.append(frame)

        return pd.concat(frames, axis=1)


def create_default_registry(*, include_jump: bool = False) -> FeatureRegistry:
    """Create a registry preloaded with Milestone 2 feature families.

    
    Parameters
    ----------
    include_jump : bool, default False
        When True, also register the opt-in ``jump`` family (M8 stretch).

    Returns
    -------
    FeatureRegistry
        Registry containing returns, har, range, and volume families,
        optionally plus ``jump``.
    """
    registry = FeatureRegistry()
    registry.register(
        FeatureSpec(
            name="returns",
            builder=build_return_features,
            description="Lagged log-return features.",
        )
    )
    registry.register(
        FeatureSpec(
            name="har",
            builder=build_har_features,
            description="HAR-style trailing realized-volatility features.",
        )
    )
    registry.register(
        FeatureSpec(
            name="range",
            builder=build_range_features,
            description="High-low range features.",
        )
    )
    registry.register(
        FeatureSpec(
            name="volume",
            builder=build_volume_features,
            description="Volume z-score features.",
        )
    )
    if include_jump:
        registry.register(
            FeatureSpec(
                name="jump",
                builder=build_jump_features,
                description=(
                    "Daily bipower-vol and jump-proportion proxies "
                    "(not tick bipower)."
                ),
            )
        )
    return registry
