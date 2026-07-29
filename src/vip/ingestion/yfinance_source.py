"""Yahoo Finance market data source adapter.

Exports
-------
YFinanceMarketDataSource
    Fetch daily OHLCV data using yfinance and return canonical frames.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import yfinance as yf

from vip.domain.errors import DataValidationError
from vip.domain.value_objects import DateRange, Symbol
from vip.ingestion.validators import validate_and_normalize_ohlcv


YAHOO_TICKER_ALIASES: dict[str, str] = {
    "VIX": "^VIX",
}


def _yahoo_ticker(symbol: Symbol) -> str:
    """Map a VIP storage symbol to the Yahoo Finance ticker."""
    return YAHOO_TICKER_ALIASES.get(symbol.value, symbol.value)


class YFinanceMarketDataSource:
    """Fetch daily OHLCV data from Yahoo Finance.

    Methods
    -------
    fetch(symbol, date_range)
        Download and normalize daily OHLCV data.
    source_name()
        Return the stable source identifier.
    """

    def fetch(self, symbol: Symbol, date_range: DateRange) -> pd.DataFrame:
        """Fetch and normalize daily OHLCV market data.

        Parameters
        ----------
        symbol : Symbol
            Instrument ticker to request.
        date_range : DateRange
            Inclusive calendar range for the request.

        Returns
        -------
        pandas.DataFrame
            Canonical normalized OHLCV frame.

        Raises
        ------
        DataValidationError
            If Yahoo Finance download fails, returns no rows,
            or data cannot be normalized/validated.
        """
        start = date_range.start.isoformat()
        # yfinance end is exclusive, so add one day for inclusive range.
        end = (date_range.end + timedelta(days=1)).isoformat()

        try:
            raw_frame = yf.download(
                _yahoo_ticker(symbol),
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                actions=False,
                multi_level_index=False,
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            raise DataValidationError(
                f"Failed to download market data from Yahoo Finance for {symbol.value}."
            ) from exc

        if raw_frame.empty:
            raise DataValidationError(
                f"No market data returned by Yahoo Finance for {symbol.value} "
                f"between {start} and {date_range.end.isoformat()}."
            )

        return validate_and_normalize_ohlcv(raw_frame)

    def source_name(self) -> str:
        """Return the stable market data source name.

        Returns
        -------
        str
            Source identifier.
        """
        return "yfinance"
