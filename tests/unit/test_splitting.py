"""Tests for walk-forward split generation."""

from __future__ import annotations

import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.evaluation.splitting import generate_expanding_folds


def test_folds_are_ordered_and_disjoint() -> None:
    """Train and test indices should be ordered, disjoint, and gap-aware."""
    index = pd.bdate_range("2020-01-01", periods=300)
    folds = generate_expanding_folds(index, n_splits=5, embargo_size=5)

    assert len(folds) == 5
    for fold in folds:
        assert fold.train_size() > 0
        assert fold.test_size() > 0
        assert fold.train_index.max() < fold.test_index.min()
        assert len(fold.train_index.intersection(fold.test_index)) == 0

        # Embargo gap: at least embargo_size labels between train end and test start.
        train_end_position = index.get_loc(fold.train_index.max())
        test_start_position = index.get_loc(fold.test_index.min())
        assert test_start_position - train_end_position - 1 >= 5


def test_expanding_train_grows() -> None:
    """Later folds should have a weakly larger training set."""
    index = pd.bdate_range("2020-01-01", periods=300)
    folds = generate_expanding_folds(index, n_splits=4, embargo_size=5)
    train_sizes = [fold.train_size() for fold in folds]
    assert train_sizes == sorted(train_sizes)


def test_unsorted_index_raises() -> None:
    """Unsorted indices should be rejected."""
    index = pd.Index([3, 1, 2])
    with pytest.raises(DataValidationError, match="sorted ascending"):
        generate_expanding_folds(index, n_splits=2, embargo_size=0)


def test_too_short_index_raises() -> None:
    """Short samples should fail rather than emit empty trains."""
    index = pd.bdate_range("2020-01-01", periods=20)
    with pytest.raises(DataValidationError):
        generate_expanding_folds(index, n_splits=5, embargo_size=5)