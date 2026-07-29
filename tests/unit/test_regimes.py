"""Tests for regime windows and sliced metrics."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from vip.evaluation.regimes import (
    RegimeWindow,
    locked_regime_windows,
    score_predictions_by_regime,
)
from vip.evaluation.walk_forward import collect_walk_forward_predictions
from vip.modeling.registry import create_default_model_registry


def _toy_predictions() -> pd.DataFrame:
    """Build OOS predictions spanning COVID and a quiet period."""
    index = pd.DatetimeIndex(
        [
            "2020-03-02",
            "2020-03-03",
            "2020-03-04",
            "2021-06-01",
            "2021-06-02",
            "2022-06-01",
            "2022-06-02",
        ]
    )
    rng = np.random.default_rng(0)
    y_true = pd.Series(0.02 + rng.normal(0, 0.001, len(index)), index=index)
    # Model A is accurate; model B is worse.
    rows = []
    for model, noise in (("good", 0.0005), ("bad", 0.01)):
        y_pred = y_true + rng.normal(0, noise, len(index))
        frame = pd.DataFrame(
            {
                "model": model,
                "fold_id": 0,
                "y_true": y_true.to_numpy(),
                "y_pred": y_pred.to_numpy(),
            },
            index=index,
        )
        rows.append(frame)
    return pd.concat(rows)


def test_locked_windows_match_research_dates() -> None:
    """Locked regimes should match the Milestone 5 table."""
    windows = {window.name: window for window in locked_regime_windows()}
    assert windows["covid_crash"].start == date(2020, 2, 20)
    assert windows["covid_crash"].end == date(2020, 4, 30)
    assert windows["bear_2022"].start == date(2022, 1, 3)
    assert windows["bear_2022"].end == date(2022, 10, 14)


def test_score_predictions_includes_full_and_named_regimes() -> None:
    """Summary should include full_sample plus locked regimes."""
    summary = score_predictions_by_regime(_toy_predictions())
    regimes = set(summary["regime"])
    assert "full_sample" in regimes
    assert "covid_crash" in regimes
    assert "bear_2022" in regimes
    covid = summary.loc[
        (summary["regime"] == "covid_crash") & (summary["model"] == "good")
    ].iloc[0]
    assert int(covid["n_obs"]) == 3


def test_empty_regime_returns_zero_obs() -> None:
    """A window with no dates should yield n_obs=0, not an exception."""
    index = pd.bdate_range("2018-01-02", periods=10)
    predictions = pd.DataFrame(
        {
            "model": ["ridge"] * len(index),
            "fold_id": 0,
            "y_true": np.full(len(index), 0.02),
            "y_pred": np.full(len(index), 0.021),
        },
        index=index,
    )
    summary = score_predictions_by_regime(
        predictions,
        regimes=(
            RegimeWindow("future", date(2030, 1, 1), date(2030, 12, 31)),
        ),
    )
    future = summary.loc[summary["regime"] == "future"].iloc[0]
    assert int(future["n_obs"]) == 0
    assert future["qlike"] is None or pd.isna(future["qlike"])


def test_collect_walk_forward_predictions_schema() -> None:
    """Collector should return dated OOS rows for each model."""
    index = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(1)
    features = pd.DataFrame(
        {"x": rng.normal(0, 1, len(index))},
        index=index,
    )
    target = pd.Series(0.05 + 0.01 * features["x"], index=index, name="y")
    registry = create_default_model_registry()
    models = registry.create_many(["historical_mean", "ridge"])
    preds = collect_walk_forward_predictions(
        features=features,
        target=target,
        models=models,
        n_splits=3,
        embargo_size=5,
    )
    assert set(preds.columns) >= {"model", "fold_id", "y_true", "y_pred"}
    assert isinstance(preds.index, pd.DatetimeIndex)
    assert set(preds["model"]) == {"historical_mean", "ridge"}