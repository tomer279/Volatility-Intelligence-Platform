"""High-low range features.

Exports
-------
build_range_features
    Build normalized range features from OHLCV.
"""

from __future__ import annotations

import pandas as pd

from vip.domain.errors import DataValidationError

REQUIRED_COLUMNS = ("high", "low", "close")
RANGE_WINDOW = 5


def build_range_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build range features available at end of session ``t``.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Canonical OHLCV frame with ``high``, ``low``, and ``close``.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - ``range_1d``: ``(high_t - low_t) / close_t``
        - ``range_5d_mean``: mean of ``range_1d`` over the last 5 sessions

    Raises
    ------
    DataValidationError
        If required columns are missing.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in ohlcv.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"OHLCV frame missing required columns: {missing_text}."
        )

    range_1d = (ohlcv["high"] - ohlcv["low"]) / ohlcv["close"]
    range_5d_mean = range_1d.rolling(window=RANGE_WINDOW, min_periods=RANGE_WINDOW).mean()
    return pd.DataFrame(
        {
            "range_1d": range_1d,
            "range_5d_mean": range_5d_mean,
        },
        index=ohlcv.index,
    )
