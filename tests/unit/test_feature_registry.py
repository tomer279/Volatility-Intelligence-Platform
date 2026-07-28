"""Tests for the feature registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.features.registry import FeatureSpec, create_default_registry


def _synthetic_ohlcv(n_rows: int = 40) -> pd.DataFrame:
    """Build synthetic canonical OHLCV data."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    close = pd.Series(np.linspace(100.0, 120.0, n_rows), index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, n_rows),
        },
        index=index,
    )


def test_default_registry_lists_expected_families() -> None:
    """Default registry should include Milestone 2 families."""
    registry = create_default_registry()
    assert registry.list_names() == ["returns", "har", "range", "volume"]


def test_build_all_concatenates_default_families() -> None:
    """Building all families should produce the expected columns."""
    registry = create_default_registry()
    features = registry.build_all(_synthetic_ohlcv())

    expected = [
        "ret_1d",
        "ret_5d",
        "rv_cc_1d",
        "rv_cc_5d",
        "rv_cc_21d",
        "range_1d",
        "range_5d_mean",
        "volume_z_21d",
    ]
    assert list(features.columns) == expected


def test_build_all_subset() -> None:
    """Registry should support building a subset of families."""
    registry = create_default_registry()
    features = registry.build_all(_synthetic_ohlcv(), names=["returns", "volume"])
    assert list(features.columns) == ["ret_1d", "ret_5d", "volume_z_21d"]


def test_unknown_family_raises() -> None:
    """Unknown family names should raise DataValidationError."""
    registry = create_default_registry()
    with pytest.raises(DataValidationError, match="Unknown feature family"):
        registry.build_all(_synthetic_ohlcv(), names=["not_a_family"])


def test_register_custom_family() -> None:
    """Custom families can be registered and built."""
    registry = create_default_registry()

    def build_dummy(ohlcv: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"dummy": 1.0}, index=ohlcv.index)

    registry.register(
        FeatureSpec(
            name="dummy",
            builder=build_dummy,
            description="Constant dummy feature.",
        )
    )
    features = registry.build_all(_synthetic_ohlcv(), names=["dummy"])
    assert list(features.columns) == ["dummy"]
    assert (features["dummy"] == 1.0).all()