"""HAR-style trailing realized-volatility features.

Exports
-------
build_har_features
    Build trailing RV features at daily, weekly, and monthly windows.
"""

from __future__ import annotations

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.features.realized import realized_volatility_trailing
from vip.features.targets import daily_log_returns

CLOSE_COLUMN = "close"
WINDOW_1D = 1
WINDOW_5D = 5
WINDOW_21D = 21


def build_har_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build HAR-style trailing realized-volatility features.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Canonical OHLCV frame with a ``close`` column.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - ``rv_cc_1d``
        - ``rv_cc_5d``
        - ``rv_cc_21d``

    Raises
    ------
    DataValidationError
        If ``close`` is missing.
    """
    if CLOSE_COLUMN not in ohlcv.columns:
        raise DataValidationError("OHLCV frame must contain a 'close' column.")

    returns = daily_log_returns(ohlcv[CLOSE_COLUMN])
    return pd.DataFrame(
        {
            "rv_cc_1d": realized_volatility_trailing(returns, WINDOW_1D),
            "rv_cc_5d": realized_volatility_trailing(returns, WINDOW_5D),
            "rv_cc_21d": realized_volatility_trailing(returns, WINDOW_21D),
        },
        index=ohlcv.index,
    )
