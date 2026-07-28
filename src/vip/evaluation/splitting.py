"""Time-series walk-forward split generation.

Exports
-------
WalkForwardFold
    One train/test fold with identifiers.
generate_expanding_folds
    Build expanding-window folds with an embargo gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vip.domain.errors import DataValidationError


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One walk-forward train/test fold.

    Parameters
    ----------
    fold_id : int
        Zero-based fold identifier.
    train_index : pandas.Index
        Training row labels.
    test_index : pandas.Index
        Test row labels.

    Methods
    -------
    train_size()
        Number of training rows.
    test_size()
        Number of test rows.
    """

    fold_id: int
    train_index: pd.Index
    test_index: pd.Index

    def train_size(self) -> int:
        """Return the number of training rows.

        Returns
        -------
        int
            Training row count.
        """
        return int(len(self.train_index))

    def test_size(self) -> int:
        """Return the number of test rows.

        Returns
        -------
        int
            Test row count.
        """
        return int(len(self.test_index))


def generate_expanding_folds(
    index: pd.Index,
    n_splits: int,
    embargo_size: int,
) -> list[WalkForwardFold]:
    """Generate expanding-window walk-forward folds with an embargo.

    The ordered index is split into ``n_splits`` contiguous test blocks of
    equal size (trailing remainder is left unused). For each test block:

    - embargo is the ``embargo_size`` labels immediately before the test start
    - train is all labels strictly before the embargo start

    Parameters
    ----------
    index : pandas.Index
        Ordered row labels (typically session dates).
    n_splits : int
        Number of test folds. Must be >= 1.
    embargo_size : int
        Number of labels excluded between train and test. Must be >= 0.

    Returns
    -------
    list of WalkForwardFold
        Generated folds in chronological order.

    Raises
    ------
    DataValidationError
        If parameters are invalid or the index is too short to form folds.
    """
    if n_splits < 1:
        raise DataValidationError("n_splits must be at least 1.")
    if embargo_size < 0:
        raise DataValidationError("embargo_size must be non-negative.")
    if not index.is_unique:
        raise DataValidationError("Index must be unique for walk-forward splits.")
    if not index.is_monotonic_increasing:
        raise DataValidationError("Index must be sorted ascending for walk-forward splits.")

    n_rows = len(index)
    test_size = n_rows // (n_splits + 1)
    if test_size < 1:
        raise DataValidationError(
            "Index is too short to create the requested walk-forward folds."
        )

    folds: list[WalkForwardFold] = []
    for fold_id in range(n_splits):
        test_end = n_rows - (n_splits - fold_id - 1) * test_size
        test_start = test_end - test_size
        embargo_start = test_start - embargo_size
        if embargo_start <= 0:
            raise DataValidationError(
                "Not enough history to form train/embargo/test for all folds. "
                "Reduce n_splits or embargo_size, or use a longer sample."
            )

        train_index = index[:embargo_start]
        test_index = index[test_start:test_end]
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_index=train_index,
                test_index=test_index,
            )
        )

    return folds
