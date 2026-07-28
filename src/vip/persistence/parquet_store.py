"""Parquet-backed storage for normalized market data.

Exports
-------
ParquetMarketDataStore
    Save/load OHLCV tables keyed by symbol.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import Symbol


class ParquetMarketDataStore:
    """Store normalized market tables as Parquet files.

    Parameters
    ----------
    root_dir : pathlib.Path
        Root directory for market data (for example ``data/raw``).

    Methods
    -------
    symbol_path(symbol)
        Return the Parquet path for a symbol.
    save(symbol, frame)
        Persist a market table.
    load(symbol)
        Load a persisted market table.
    exists(symbol)
        Return whether a table exists for ``symbol``.
    """

    def __init__(self, root_dir: Path) -> None:
        """Initialize the store.

        Parameters
        ----------
        root_dir : pathlib.Path
            Root directory under which per-symbol folders are created.
        """
        self._root_dir = root_dir

    def symbol_path(self, symbol: Symbol) -> Path:
        """Return the Parquet path for a symbol.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.

        Returns
        -------
        pathlib.Path
            Path of the form ``{root}/{SYMBOL}/ohlcv.parquet``.
        """
        return self._root_dir / symbol.as_path_key() / "ohlcv.parquet"

    def save(self, symbol: Symbol, frame: pd.DataFrame) -> Path:
        """Persist a market data table.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.
        frame : pandas.DataFrame
            Normalized OHLCV table.

        Returns
        -------
        pathlib.Path
            Path written to disk.

        Raises
        ------
        PersistenceError
            If the frame cannot be written.
        """
        path = self.symbol_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(path)
        except (OSError, ValueError, TypeError) as exc:
            raise PersistenceError(
                f"Failed to save market data for {symbol.value}: {exc}"
            ) from exc
        return path

    def load(self, symbol: Symbol) -> pd.DataFrame:
        """Load a persisted market data table.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.

        Returns
        -------
        pandas.DataFrame
            Normalized OHLCV table.

        Raises
        ------
        PersistenceError
            If the file is missing or unreadable.
        """
        path = self.symbol_path(symbol)
        if not path.is_file():
            raise PersistenceError(
                f"No market data found for {symbol.value} at {path}"
            )
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError, TypeError) as exc:
            raise PersistenceError(
                f"Failed to load market data for {symbol.value}: {exc}"
            ) from exc

    def exists(self, symbol: Symbol) -> bool:
        """Return whether data for a symbol exists on disk.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.

        Returns
        -------
        bool
            True when the Parquet file is present.
        """
        return self.symbol_path(symbol).is_file()
