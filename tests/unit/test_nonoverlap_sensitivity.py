"""Unit tests for non-overlapping OOS sensitivity thinning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.evaluation.inference import (
    BootstrapInferenceOptions,
    non_overlapping_index,
    non_overlapping_subsample,
    block_bootstrap_nonoverlap_sensitivity,
)


HORIZON_DAYS = 5
N_DATES = 40


def test_non_overlapping_index_every_fifth_business_day() -> None:
    """Stride must keep positions 0, 5, 10, ... on sorted unique dates."""
    index = pd.bdate_range("2020-01-01", periods=N_DATES)
    thinned = non_overlapping_index(index, HORIZON_DAYS)
    expected = index[::HORIZON_DAYS]
    assert list(thinned) == list(expected)
    assert len(thinned) == (N_DATES + HORIZON_DAYS - 1) // HORIZON_DAYS


def test_non_overlapping_subsample_preserves_values() -> None:
    """Thinned series values must match the strided source."""
    index = pd.bdate_range("2020-01-01", periods=N_DATES)
    values = pd.Series(np.arange(N_DATES, dtype=float), index=index)
    thinned = non_overlapping_subsample(values, HORIZON_DAYS)
    assert thinned.tolist() == values.iloc[::HORIZON_DAYS].tolist()


def test_non_overlapping_rejects_bad_horizon() -> None:
    """horizon_days < 1 must raise."""
    index = pd.bdate_range("2020-01-01", periods=10)
    with pytest.raises(DataValidationError):
        non_overlapping_index(index, 0)


def test_sensitivity_skips_when_too_short_for_block() -> None:
    """Thinned length below block_length yields skipped_too_short."""
    index = pd.bdate_range("2020-01-01", periods=20)
    differential = pd.Series(np.linspace(-0.1, 0.1, 20), index=index)
    options = BootstrapInferenceOptions(
        block_length=10,
        n_resamples=99,
        alpha=0.05,
        random_seed=0,
    )
    # stride 5 → 4 obs < block_length 10
    bootstrap, n_full, n_thinned, status = block_bootstrap_nonoverlap_sensitivity(
        differential,
        horizon_days=5,
        options=options,
    )
    assert bootstrap is None
    assert n_full == 20
    assert n_thinned == 4
    assert status == "skipped_too_short"


def test_sensitivity_ok_on_long_series() -> None:
    """Long overlapping series thins enough to bootstrap."""
    index = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    differential = pd.Series(-0.2 + rng.normal(0.0, 0.05, 200), index=index)
    options = BootstrapInferenceOptions(
        block_length=10,
        n_resamples=199,
        alpha=0.05,
        random_seed=0,
    )
    bootstrap, n_full, n_thinned, status = block_bootstrap_nonoverlap_sensitivity(
        differential,
        horizon_days=5,
        options=options,
    )
    assert status == "ok"
    assert n_full == 200
    assert n_thinned == 40
    assert bootstrap is not None
    assert bootstrap.ci_low <= bootstrap.mean_delta <= bootstrap.ci_high