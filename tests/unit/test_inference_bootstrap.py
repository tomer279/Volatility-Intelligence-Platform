"""Correctness tests for block-bootstrap inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.domain.errors import DataValidationError
from vip.evaluation.inference import (
    BootstrapInferenceOptions,
    block_bootstrap_mean,
    loss_differential,
    nw_lags_for_horizon,
)


def _ar1_series(
    n_obs: int,
    ar_coef: float,
    mean: float,
    seed: int,
) -> pd.Series:
    """Simulate a mild AR(1) loss-differential series."""
    rng = np.random.default_rng(seed)
    noise_scale = 1.0
    values = np.empty(n_obs, dtype=float)
    values[0] = mean + rng.normal(0.0, noise_scale)
    for index in range(1, n_obs):
        values[index] = (
            mean * (1.0 - ar_coef)
            + ar_coef * values[index - 1]
            + rng.normal(0.0, noise_scale)
        )
    return pd.Series(values)


def _iid_bootstrap_mean_ci_width(
    values: np.ndarray,
    n_resamples: int,
    alpha: float,
    seed: int,
) -> float:
    """Percentile CI width under i.i.d. day resampling (test helper only)."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for resample_index in range(n_resamples):
        sample = rng.choice(values, size=values.shape[0], replace=True)
        means[resample_index] = float(np.mean(sample))
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return upper - lower


def test_nw_lags_for_horizon_default_target() -> None:
    """5-day target must use NW lags = 4."""
    assert nw_lags_for_horizon(5) == 4


def test_loss_differential_alignment() -> None:
    """Differentials should inner-join on the shared index."""
    challenger = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    baseline = pd.Series([1.5, 1.5], index=[1, 2])
    differential = loss_differential(challenger, baseline)
    assert list(differential.index) == [1, 2]
    assert differential.tolist() == pytest.approx([0.5, 1.5])


def test_bootstrap_rejects_under_alternative() -> None:
    """Large negative mean should reject H0 at alpha=0.05."""
    differential = _ar1_series(
        n_obs=200,
        ar_coef=0.3,
        mean=-0.5,
        seed=7,
    )
    options = BootstrapInferenceOptions(
        block_length=15,
        n_resamples=399,
        alpha=0.05,
        random_seed=0,
    )
    result = block_bootstrap_mean(differential, options)
    assert result.mean_delta < 0.0
    assert result.pvalue <= 0.05
    assert result.rejects_null(0.05)


def test_bootstrap_null_rejection_rate_near_alpha() -> None:
    """Under a mean-zero AR(1), rejection rate should stay near alpha.

    Assertion (documented): with modest Monte Carlo size, empirical
    rejection rate at alpha=0.05 lies in [0.0, 0.20].
    """
    alpha = 0.05
    n_trials = 40
    options = BootstrapInferenceOptions(
        block_length=12,
        n_resamples=199,
        alpha=alpha,
        random_seed=0,
    )
    rejections = 0
    for trial in range(n_trials):
        differential = _ar1_series(
            n_obs=180,
            ar_coef=0.4,
            mean=0.0,
            seed=1000 + trial,
        )
        # Re-seed options per trial via a fresh options object
        trial_options = BootstrapInferenceOptions(
            block_length=options.block_length,
            n_resamples=options.n_resamples,
            alpha=options.alpha,
            random_seed=trial,
        )
        result = block_bootstrap_mean(differential, trial_options)
        if result.rejects_null(alpha):
            rejections += 1
    rate = rejections / n_trials
    assert 0.0 <= rate <= 0.20


def test_block_ci_wider_than_iid_under_dependence() -> None:
    """Under strong serial dependence, block CI should be wider than iid.

    Assertion (documented): for an MA-like overlapping series, the
    block-bootstrap percentile CI width exceeds the i.i.d. bootstrap
    CI width (iid understates dependence).
    """
    rng = np.random.default_rng(11)
    innovations = rng.normal(0.0, 1.0, size=250)
    # Overlapping-style dependence: d_t = e_t + e_{t-1} + ... + e_{t-4}
    window = 5
    values = np.convolve(innovations, np.ones(window), mode="valid")
    differential = pd.Series(values)
    options = BootstrapInferenceOptions(
        block_length=15,
        n_resamples=799,
        alpha=0.05,
        random_seed=0,
    )
    result = block_bootstrap_mean(differential, options)
    block_width = result.ci_high - result.ci_low
    iid_width = _iid_bootstrap_mean_ci_width(
        values=values,
        n_resamples=options.n_resamples,
        alpha=options.alpha,
        seed=0,
    )
    assert block_width > iid_width


def test_invalid_block_length_raises() -> None:
    """Block length outside 10–20 should raise."""
    options = BootstrapInferenceOptions(block_length=5)
    with pytest.raises(DataValidationError, match="block_length must be in"):
        options.validate()