"""IV−RV gap features from as-of VIX level and trailing HAR RV.

Exports
-------
vix_level_to_daily_vol
    Convert annualized VIX percent prints to non-annualized daily vol.
build_iv_rv_features
    Build ``vix_vol_daily``, IV−RV gaps at HAR windows, and optional ratio.
WINDOW_1D, WINDOW_5D, WINDOW_21D
    Trailing window lengths aligned with HAR (1 / 5 / 21).
ANNUALIZATION_DAYS
    Trading-day count used in the locked VIX → daily conversion.
PERCENT_TO_FRACTION
    Divisor mapping VIX percent prints to decimal annual vol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vip.domain.errors import DataValidationError

WINDOW_1D = 1
WINDOW_5D = 5
WINDOW_21D = 21
ANNUALIZATION_DAYS = 252
PERCENT_TO_FRACTION = 100.0

RV_CC_1D_COLUMN = f"rv_cc_{WINDOW_1D}d"
RV_CC_5D_COLUMN = f"rv_cc_{WINDOW_5D}d"
RV_CC_21D_COLUMN = f"rv_cc_{WINDOW_21D}d"
REQUIRED_HAR_COLUMNS = (RV_CC_1D_COLUMN, RV_CC_5D_COLUMN, RV_CC_21D_COLUMN)

VIX_VOL_DAILY_COLUMN = "vix_vol_daily"
VIX_MINUS_RV_1D_COLUMN = f"vix_minus_rv_{WINDOW_1D}d"
VIX_MINUS_RV_5D_COLUMN = f"vix_minus_rv_{WINDOW_5D}d"
VIX_MINUS_RV_21D_COLUMN = f"vix_minus_rv_{WINDOW_21D}d"
VIX_RV_RATIO_5D_COLUMN = f"vix_rv_ratio_{WINDOW_5D}d"


def vix_level_to_daily_vol(vix_level: pd.Series) -> pd.Series:
    """Convert VIX percent prints to non-annualized daily volatility.

    Locked research conversion::

        vix_vol_daily = (vix_level / 100) / sqrt(252)

    Parameters
    ----------
    vix_level : pandas.Series
        As-of VIX close levels (annualized percent), indexed by session.

    Returns
    -------
    pandas.Series
        Daily-vol-scale series named ``vix_vol_daily``. Does not mutate
        ``vix_level``.
    """
    daily = (vix_level / PERCENT_TO_FRACTION) / np.sqrt(ANNUALIZATION_DAYS)
    daily = daily.rename(VIX_VOL_DAILY_COLUMN)
    return daily


def build_iv_rv_features(
        har_frame: pd.DataFrame,
        vix_level: pd.Series,
) -> pd.DataFrame:
    """Build IV−RV gap features from trailing HAR RV and as-of VIX.

    Uses information available at session ``t`` only: ``vix_level`` at ``t``
    (caller must as-of-align) and trailing ``rv_cc_*`` ending at ``t``.
    Does not read ``target_rv_cc_*`` columns. Does not mutate inputs.

    Parameters
    ----------
    har_frame : pandas.DataFrame
        Frame containing ``rv_cc_1d``, ``rv_cc_5d``, and ``rv_cc_21d``.
    vix_level : pandas.Series
        VIX levels already aligned to ``har_frame.index`` (pipeline as-of
        join). Values are reindexed to that index; missing labels → NaN.

    Returns
    -------
    pandas.DataFrame
        Columns ``vix_vol_daily``, ``vix_minus_rv_1d``, ``vix_minus_rv_5d``,
        ``vix_minus_rv_21d``, and ``vix_rv_ratio_5d`` (NaN where
        ``rv_cc_5d`` is non-positive or missing).

    Raises
    ------
    DataValidationError
        If any required HAR RV column is missing.
    """
    _require_har_columns(har_frame)

    vix_aligned = vix_level.reindex(har_frame.index)
    vix_vol_daily = vix_level_to_daily_vol(vix_aligned)
    rv_1d = har_frame[RV_CC_1D_COLUMN]
    rv_5d = har_frame[RV_CC_5D_COLUMN]
    rv_21d = har_frame[RV_CC_21D_COLUMN]
    ratio_5d = vix_vol_daily.where(rv_5d > 0.0) / rv_5d.where(rv_5d > 0.0)

    return pd.DataFrame(
        {
            VIX_VOL_DAILY_COLUMN: vix_vol_daily,
            VIX_MINUS_RV_1D_COLUMN: vix_vol_daily - rv_1d,
            VIX_MINUS_RV_5D_COLUMN: vix_vol_daily - rv_5d,
            VIX_MINUS_RV_21D_COLUMN: vix_vol_daily - rv_21d,
            VIX_RV_RATIO_5D_COLUMN: ratio_5d,
        },
        index=har_frame.index,
    )


def _require_har_columns(har_frame: pd.DataFrame) -> None:
    """Raise when required trailing HAR RV columns are absent."""
    missing = [name for name in REQUIRED_HAR_COLUMNS if name not in har_frame.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise DataValidationError(
            f"HAR frame missing required columns: {missing_list}."
        )
