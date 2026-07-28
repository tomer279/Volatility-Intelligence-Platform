"""Tests for volatility forecast metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.evaluation.metrics import mae, mse, qlike


def test_mse_and_mae_hand_values() -> None:
    """MSE/MAE should match direct calculations."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 1.0, 5.0])

    assert mse(y_true, y_pred) == pytest.approx(
        np.mean([(0.0) ** 2, (1.0) ** 2, (-2.0) ** 2])
    )
    assert mae(y_true, y_pred) == pytest.approx(np.mean([0.0, 1.0, 2.0]))


def test_qlike_hand_value() -> None:
    """QLIKE should match the documented formula."""
    y_true = pd.Series([0.2, 0.3])
    y_pred = pd.Series([0.25, 0.25])
    expected = np.mean(
        [
            np.log(0.25**2) + (0.2**2) / (0.25**2),
            np.log(0.25**2) + (0.3**2) / (0.25**2),
        ]
    )
    assert qlike(y_true, y_pred) == pytest.approx(expected)


def test_qlike_clips_non_positive_predictions() -> None:
    """Non-positive forecasts should be clipped by epsilon."""
    y_true = pd.Series([0.2])
    y_pred = pd.Series([0.0])
    epsilon = 1e-6
    expected = np.log(epsilon**2) + (0.2**2) / (epsilon**2)
    assert qlike(y_true, y_pred, epsilon=epsilon) == pytest.approx(expected)


def test_empty_alignment_raises() -> None:
    """Metrics should raise when no overlapping values remain."""
    y_true = pd.Series([1.0], index=["a"])
    y_pred = pd.Series([1.0], index=["b"])
    with pytest.raises(DataValidationError, match="No overlapping finite"):
        mse(y_true, y_pred)


def test_invalid_epsilon_raises() -> None:
    """Non-positive epsilon should raise DataValidationError."""
    y_true = pd.Series([0.2])
    y_pred = pd.Series([0.2])
    with pytest.raises(DataValidationError, match="epsilon must be positive"):
        qlike(y_true, y_pred, epsilon=0.0)