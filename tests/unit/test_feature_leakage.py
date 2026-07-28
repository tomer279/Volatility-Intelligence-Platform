"""Leakage and temporal-alignment tests for feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.features.pipeline import build_feature_matrix
from vip.features.registry import create_default_registry
from vip.features.targets import build_target_rv_cc


def _synthetic_ohlcv(n_rows: int = 80) -> pd.DataFrame:
    """Build synthetic canonical-like OHLCV data."""
    index = pd.bdate_range("2024-01-02", periods=n_rows)
    close = pd.Series(np.linspace(100.0, 140.0, n_rows), index=index)
    return pd.DataFrame(
        {
            "open": close.to_numpy(),
            "high": (close + 1.0).to_numpy(),
            "low": (close - 1.0).to_numpy(),
            "close": close.to_numpy(),
            "volume": np.linspace(1_000.0, 3_000.0, n_rows),
        },
        index=index,
    )


def test_features_at_t_unchanged_when_future_is_truncated() -> None:
    """Features at t must depend only on data through t."""
    full = _synthetic_ohlcv()
    registry = create_default_registry()
    full_features = registry.build_all(full)

    cutoff_position = 50
    cutoff_date = full.index[cutoff_position]
    truncated = full.iloc[: cutoff_position + 1]
    truncated_features = registry.build_all(truncated)

    feature_columns = list(full_features.columns)
    pd.testing.assert_series_equal(
        full_features.loc[cutoff_date, feature_columns],
        truncated_features.loc[cutoff_date, feature_columns],
        check_names=False,
    )


def test_future_close_shock_changes_target_not_features_at_t() -> None:
    """A shock after t should affect target at t, not features at t."""
    base = _synthetic_ohlcv()
    registry = create_default_registry()

    t_position = 40
    t_date = base.index[t_position]
    shock_date = base.index[t_position + 2]

    shocked = base.copy()
    shocked.loc[shock_date, "close"] = shocked.loc[shock_date, "close"] * 1.20
    shocked.loc[shock_date, "open"] = shocked.loc[shock_date, "close"]
    shocked.loc[shock_date, "high"] = shocked.loc[shock_date, "close"] + 1.0
    shocked.loc[shock_date, "low"] = shocked.loc[shock_date, "close"] - 1.0

    base_features = registry.build_all(base)
    shocked_features = registry.build_all(shocked)
    pd.testing.assert_series_equal(
        base_features.loc[t_date],
        shocked_features.loc[t_date],
        check_names=False,
    )

    base_target = build_target_rv_cc(base, horizon_days=5)
    shocked_target = build_target_rv_cc(shocked, horizon_days=5)
    assert base_target.loc[t_date] != pytest.approx(shocked_target.loc[t_date])


def test_pipeline_output_has_no_nans() -> None:
    """Saved-style matrix from the pipeline must be fully observed."""
    matrix = build_feature_matrix(_synthetic_ohlcv(), horizon_days=5)
    assert not matrix.isna().any().any()
    assert "target_rv_cc_5d" in matrix.columns