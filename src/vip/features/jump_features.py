"""Jump-robust daily realized-feature family.

Exports
-------
build_jump_features
    Build trailing bipower-vol and jump-proportion columns at HAR windows.
"""

from __future__ import annotations

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.features.realized import (
    bipower_volatility_trailing,
    jump_proportion_trailing,
)
from vip.features.targets import daily_log_returns

CLOSE_COLUMN = "close"
WINDOW_1D = 1
WINDOW_5D = 5
WINDOW_21D = 21


def build_jump_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build daily jump-robust features available at end of session ``t``.

    Columns use trailing windows only (information set ``<= t``). Estimators
    are close-to-close daily proxies, not high-frequency bipower.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Canonical OHLCV frame with a ``close`` column.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - ``bpv_cc_1d``, ``bpv_cc_5d``, ``bpv_cc_21d``
        - ``jump_prop_1d``, ``jump_prop_5d``, ``jump_prop_21d``

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
            "bpv_cc_1d": bipower_volatility_trailing(returns, WINDOW_1D),
            "bpv_cc_5d": bipower_volatility_trailing(returns, WINDOW_5D),
            "bpv_cc_21d": bipower_volatility_trailing(returns, WINDOW_21D),
            "jump_prop_1d": jump_proportion_trailing(returns, WINDOW_1D),
            "jump_prop_5d": jump_proportion_trailing(returns, WINDOW_5D),
            "jump_prop_21d": jump_proportion_trailing(returns, WINDOW_21D),
        },
        index=ohlcv.index,
    )
