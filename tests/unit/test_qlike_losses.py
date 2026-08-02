"""Tests for elementwise QLIKE losses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.evaluation.metrics import qlike, qlike_losses


def test_qlike_losses_mean_matches_scalar() -> None:
    """Elementwise mean should match scalar qlike within float tolerance."""
    y_true = pd.Series([0.2, 0.3, 0.25], index=pd.RangeIndex(3))
    y_pred = pd.Series([0.22, 0.28, 0.24], index=pd.RangeIndex(3))
    losses = qlike_losses(y_true, y_pred)
    assert float(losses.mean()) == pytest.approx(qlike(y_true, y_pred))


def test_qlike_losses_applies_epsilon_floor() -> None:
    """Non-positive forecasts should be floored by epsilon."""
    y_true = pd.Series([0.2])
    y_pred = pd.Series([0.0])
    epsilon = 1e-6
    expected = np.log(epsilon**2) + (0.2**2) / (epsilon**2)
    assert float(qlike_losses(y_true, y_pred, epsilon=epsilon).iloc[0]) == pytest.approx(
        expected
    )


def test_qlike_losses_misaligned_raises() -> None:
    """Empty overlap should raise DataValidationError."""
    y_true = pd.Series([1.0], index=["a"])
    y_pred = pd.Series([1.0], index=["b"])
    with pytest.raises(DataValidationError, match="No overlapping finite"):
        qlike_losses(y_true, y_pred)


def test_qlike_losses_invalid_epsilon_raises() -> None:
    """Non-positive epsilon should raise DataValidationError."""
    with pytest.raises(DataValidationError, match="epsilon must be positive"):
        qlike_losses(pd.Series([0.2]), pd.Series([0.2]), epsilon=0.0)