"""End-to-end feature-matrix construction.

Exports
-------
build_feature_matrix
    Build own-symbol features and target; optionally join VIX columns, then
    drop incomplete rows.
"""

from __future__ import annotations

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.features.registry import FeatureRegistry, create_default_registry
from vip.features.targets import build_target_rv_cc
from vip.ingestion.validators import validate_and_normalize_ohlcv
from vip.features.cross_asset import build_vix_features

DEFAULT_HORIZON_DAYS = 5


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    feature_names: list[str] | None = None,
    registry: FeatureRegistry | None = None,
    vix_ohlcv: pd.DataFrame | None = None,
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
    vix_ohlcv : pandas.DataFrame or None, default None
        Optional VIX OHLCV used to append ``vix_level`` / ``vix_chg_1d``.

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

    canonical = validate_and_normalize_ohlcv(ohlcv)
    active_registry = registry if registry is not None else create_default_registry()
    features = active_registry.build_all(canonical, names=feature_names)
    pieces: list[pd.DataFrame | pd.Series] = [features]

    if vix_ohlcv is not None:
        pieces.append(build_vix_features(canonical.index, vix_ohlcv))

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
