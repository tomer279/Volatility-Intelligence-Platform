"""Return-based predictive features.

Exports
-------
build_return_features
    Build lagged return features from canonical OHLCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vip.domain.errors import DataValidationError
from vip.features.targets import daily_log_returns

CLOSE_COLUMN = "close"


def build_return_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build return features available at end of session ``t``.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Canonical OHLCV frame with a ``close`` column.

    Returns
    -------
    pandas.DataFrame
        Columns ``ret_1d`` and ``ret_5d``.

    Raises
    ------
    DataValidationError
        If ``close`` is missing.
    """
    if CLOSE_COLUMN not in ohlcv.columns:
        raise DataValidationError("OHLCV frame must contain a 'close' column.")

    close = ohlcv[CLOSE_COLUMN]
    return pd.DataFrame(
        {
            "ret_1d": daily_log_returns(close),
            "ret_5d": np.log(close / close.shift(5)),
        },
        index=ohlcv.index,
    )
