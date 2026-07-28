"""Parquet-backed storage for feature matrices.

Exports
-------
ParquetFeatureMatrixStore
    Save/load feature matrices keyed by symbol.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import Symbol


class ParquetFeatureMatrixStore:
    """Store feature matrices as Parquet files.

    Parameters
    ----------
    root_dir : pathlib.Path
        Root directory for processed features (for example ``data/processed``).

    Methods
    -------
    symbol_path(symbol)
        Return the Parquet path for a symbol.
    save(symbol, frame)
        Persist a feature matrix.
    load(symbol)
        Load a persisted feature matrix.
    exists(symbol)
        Return whether a matrix exists for ``symbol``.
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
        """Return the feature-matrix Parquet path for a symbol.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.

        Returns
        -------
        pathlib.Path
            Path of the form ``{root}/{SYMBOL}/features.parquet``.
        """
        return self._root_dir / symbol.as_path_key() / "features.parquet"

    def save(self, symbol: Symbol, frame: pd.DataFrame) -> Path:
        """Persist a feature matrix.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.
        frame : pandas.DataFrame
            Feature matrix including target column.

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
                f"Failed to save feature matrix for {symbol.value}: {exc}"
            ) from exc
        return path

    def load(self, symbol: Symbol) -> pd.DataFrame:
        """Load a persisted feature matrix.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.

        Returns
        -------
        pandas.DataFrame
            Feature matrix.

        Raises
        ------
        PersistenceError
            If the file is missing or unreadable.
        """
        path = self.symbol_path(symbol)
        if not path.is_file():
            raise PersistenceError(
                f"No feature matrix found for {symbol.value} at {path}"
            )
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError, TypeError) as exc:
            raise PersistenceError(
                f"Failed to load feature matrix for {symbol.value}: {exc}"
            ) from exc

    def exists(self, symbol: Symbol) -> bool:
        """Return whether a feature matrix exists for a symbol.

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
