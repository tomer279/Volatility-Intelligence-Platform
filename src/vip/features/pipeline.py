"""End-to-end feature-matrix construction.

Exports
-------
VixJoinOptions
    Optional VIX / IV−RV / rates joins for the pipeline.
build_feature_matrix
    Build own-symbol features and target; optionally join VIX columns,
    optionally append IV−RV gaps and/or rates columns, then drop incomplete rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.features.cross_asset import (
    VIX_LEVEL_COLUMN,
    build_vix_features,
    build_rates_features
)
from vip.features.iv_rv_features import build_iv_rv_features
from vip.features.registry import FeatureRegistry, create_default_registry
from vip.features.targets import build_target_rv_cc
from vip.ingestion.validators import validate_and_normalize_ohlcv

DEFAULT_HORIZON_DAYS = 5


@dataclass(frozen=True, slots=True)
class VixJoinOptions:
    """Optional VIX / IV−RV / rates joins for ``build_feature_matrix``.

    Parameters
    ----------
    vix_ohlcv : pandas.DataFrame or None, default None
        VIX OHLCV for as-of ``vix_level`` / ``vix_chg_1d``.
    include_iv_rv : bool, default False
        When True, append IV−RV columns via ``build_iv_rv_features`` after
        HAR features and as-of ``vix_level`` exist.
    rates_ohlcv : pandas.DataFrame or None, default None
        TNX OHLCV for as-of ``tnx_level`` / ``tnx_chg_1d``.

    Methods
    -------
    describe()
        Return a short human-readable summary.
    has_vix()
        Return whether VIX OHLCV is attached.
    has_rates()
        Return whether rates OHLCV is attached.
    requires_iv_rv()
        Return whether IV−RV columns should be appended.
    """

    vix_ohlcv: pd.DataFrame | None = None
    include_iv_rv: bool = False
    rates_ohlcv: pd.DataFrame | None = None

    def describe(self) -> str:
        """Return a short human-readable summary.
    
        Returns
        -------
        str
            Compact join-options summary.
        """
        return (
            f"has_vix={self.has_vix()}, "
            f"include_iv_rv={self.include_iv_rv}, "
            f"has_rates={self.has_rates()}"
        )

    def has_vix(self) -> bool:
        """Return whether VIX OHLCV is attached.
    
        Returns
        -------
        bool
            True when ``vix_ohlcv`` is not None.
        """
        return self.vix_ohlcv is not None

    def has_rates(self) -> bool:
        """Return whether rates OHLCV is attached.
    
        Returns
        -------
        bool
            True when ``rates_ohlcv`` is not None.
        """
        return self.rates_ohlcv is not None

    def requires_iv_rv(self) -> bool:
        """Return whether IV−RV columns should be appended.
    
        Returns
        -------
        bool
            True when ``include_iv_rv`` is set.
        """
        return self.include_iv_rv


def build_feature_matrix(
        ohlcv: pd.DataFrame,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        feature_names: list[str] | None = None,
        registry: FeatureRegistry | None = None,
        vix_ohlcv: pd.DataFrame | VixJoinOptions | None = None,
) -> pd.DataFrame:
    """Build a clean feature matrix with forward RV target.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Daily OHLCV input. Validated/normalized before feature construction.
    horizon_days : int, default 5
        Forward target horizon in trading days.
    feature_names : list of str or None, default None
        Feature-family names to include. ``None`` uses all registered families.
    registry : FeatureRegistry or None, default None
        Feature registry. ``None`` uses ``create_default_registry()``.
    vix_ohlcv : pandas.DataFrame, VixJoinOptions, or None, default None
        Optional VIX input. Pass a DataFrame for level/chg only (legacy),
        or ``VixJoinOptions`` for IV−RV and/or rates joins.

    Returns
    -------
    pandas.DataFrame
        Feature columns plus ``target_rv_cc_{horizon_days}d``, with rows
        containing any NaN removed.

    Raises
    ------
    DataValidationError
        If validation fails, no rows remain after cleaning, or inputs are invalid.
    """
    if horizon_days < 1:
        raise DataValidationError("Horizon must be at least 1 trading day.")

    vix_options = _normalize_vix_input(vix_ohlcv)

    canonical = validate_and_normalize_ohlcv(ohlcv)
    active_registry = registry if registry is not None else create_default_registry()
    features = active_registry.build_all(canonical, names=feature_names)
    pieces: list[pd.DataFrame | pd.Series] = [features]

    if vix_options is not None:
        pieces.extend(
            _cross_asset_feature_frames(
                features,
                canonical.index,
                vix_options,
            )
        )

    target = build_target_rv_cc(canonical, horizon_days=horizon_days)
    pieces.append(target.rename(target.name))

    matrix = pd.concat(pieces, axis=1)
    cleaned = matrix.dropna(axis=0, how="any")

    if cleaned.empty:
        raise DataValidationError(
            "Feature matrix is empty after dropping incomplete rows. "
            "Use a longer OHLCV history."
        )

    return cleaned


def _cross_asset_feature_frames(
        har_features: pd.DataFrame,
        primary_index: pd.DatetimeIndex,
        options: VixJoinOptions,
) -> list[pd.DataFrame]:
    """Build optional VIX, IV−RV, and rates frames for the pipeline.

    Parameters
    ----------
    har_features : pandas.DataFrame
        Own-symbol feature block (must include HAR ``rv_cc_*`` when IV−RV
        is requested).
    primary_index : pandas.DatetimeIndex
        Primary session calendar.
    options : VixJoinOptions
        Attached VIX / rates OHLCV and IV−RV toggle.

    Returns
    -------
    list of pandas.DataFrame
        Zero or more frames to concatenate beside own-symbol features.

    Raises
    ------
    DataValidationError
        If IV−RV is requested without VIX OHLCV.
    """
    if options.requires_iv_rv() and not options.has_vix():
        raise DataValidationError(
            "include_iv_rv requires VIX OHLCV on VixJoinOptions."
        )

    frames: list[pd.DataFrame] = []
    vix_frame: pd.DataFrame | None = None

    if options.has_vix():
        vix_frame = build_vix_features(primary_index, options.vix_ohlcv)
        frames.append(vix_frame)

    if options.requires_iv_rv():
        # has_vix was validated above; vix_frame is non-None here.
        frames.append(
            build_iv_rv_features(
                har_features,
                vix_frame[VIX_LEVEL_COLUMN],
            )
        )

    if options.has_rates():
        frames.append(
            build_rates_features(primary_index, options.rates_ohlcv)
        )

    return frames


def _normalize_vix_input(
        vix_ohlcv: pd.DataFrame | VixJoinOptions | None,
) -> VixJoinOptions | None:
    """Normalize legacy DataFrame or ``VixJoinOptions`` to one options object."""
    if vix_ohlcv is None:
        return None
    if isinstance(vix_ohlcv, VixJoinOptions):
        return vix_ohlcv
    return VixJoinOptions(vix_ohlcv=vix_ohlcv, include_iv_rv=False)
