"""Cross-asset features joined onto a primary session calendar.

Exports
-------
build_vix_features
    Align VIX level and 1-day change to a primary DatetimeIndex.
"""

from __future__ import annotations

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.ingestion.validators import validate_and_normalize_ohlcv

CLOSE_COLUMN = "close"
VIX_LEVEL_COLUMN = "vix_level"
VIX_CHG_COLUMN = "vix_chg_1d"


def build_vix_features(
    primary_index: pd.DatetimeIndex,
    vix_ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    """Build VIX features available at each primary session ``t``.

    VIX prints are aligned with ``merge_asof(..., direction="backward")``,
    so each primary date uses the latest VIX close with timestamp ``<= t``.

    Parameters
    ----------
    primary_index : pandas.DatetimeIndex
        Session calendar of the primary symbol.
    vix_ohlcv : pandas.DataFrame
        VIX OHLCV (validated/normalized inside this function).

    Returns
    -------
    pandas.DataFrame
        Columns ``vix_level`` and ``vix_chg_1d`` indexed like
        ``primary_index``. Early rows may be NaN until the first VIX print.

    Raises
    ------
    DataValidationError
        If ``primary_index`` is empty or VIX data is invalid.
    """
    if len(primary_index) == 0:
        raise DataValidationError("primary_index must be non-empty.")

    canonical = validate_and_normalize_ohlcv(vix_ohlcv)
    close = canonical[CLOSE_COLUMN]
    vix_frame = pd.DataFrame(
        {
            VIX_LEVEL_COLUMN: close,
            VIX_CHG_COLUMN: close.pct_change(),
        },
        index=canonical.index,
    )
    return _asof_align(primary_index, vix_frame)


def _asof_align(
    primary_index: pd.DatetimeIndex,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Left-asof-join ``feature_frame`` onto ``primary_index`` (no lookahead)."""
    left = pd.DataFrame({"_session": pd.DatetimeIndex(primary_index)}).sort_values(
        "_session"
    )
    right = feature_frame.copy()
    right["_session"] = right.index
    right = right.sort_values("_session")

    merged = pd.merge_asof(
        left,
        right,
        on="_session",
        direction="backward",
    )
    merged = merged.set_index("_session")
    merged.index.name = primary_index.name
    return merged.loc[:, list(feature_frame.columns)]
