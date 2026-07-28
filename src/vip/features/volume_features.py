"""Volume-based predictive features.

Exports
-------
build_volume_features
    Build volume z-score features from OHLCV.
"""

from __future__ import annotations

import pandas as pd

from vip.domain.errors import DataValidationError

VOLUME_COLUMN = "volume"
VOLUME_WINDOW = 21


def build_volume_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build volume features available at end of session ``t``.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Canonical OHLCV frame with a ``volume`` column.

    Returns
    -------
    pandas.DataFrame
        Column ``volume_z_21d``:
        ``(volume_t - mean_21) / std_21`` using a trailing window ending at ``t``.

    Raises
    ------
    DataValidationError
        If ``volume`` is missing.
    """
    if VOLUME_COLUMN not in ohlcv.columns:
        raise DataValidationError("OHLCV frame must contain a 'volume' column.")

    volume = ohlcv[VOLUME_COLUMN]
    mean_21 = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()
    std_21 = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).std()
    volume_z = (volume - mean_21) / std_21
    return pd.DataFrame({"volume_z_21d": volume_z}, index=ohlcv.index)
