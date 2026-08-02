"""Statistical inference for out-of-sample loss differentials.

Exports
-------
BootstrapInferenceOptions
    Block-bootstrap settings (length, resamples, alpha, seed).
BootstrapResult
    Mean gap, percentile CI, and two-sided bootstrap p-value.
DMResult
    Diebold–Mariano statistic with HLN correction and p-value.
nw_lags_for_horizon
    Newey–West lag locked to ``horizon_days - 1``.
loss_differential
    Aligned challenger minus baseline per-row losses.
block_bootstrap_mean
    Moving block bootstrap of the mean loss differential.
hln_diebold_mariano
    DM test with Newey–West HAC and HLN finite-sample correction.
NonOverlapSensitivityResult
    Footnote bootstrap result on a horizon-strided subsample.
non_overlapping_index
    Keep every horizon_days-th label from a sorted unique index.
non_overlapping_subsample
    Thin a time-ordered series to non-overlapping horizon spacing.
block_bootstrap_nonoverlap_sensitivity
    Block-bootstrap mean(d) on the non-overlapping subsample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from vip.domain.errors import DataValidationError

DEFAULT_BLOCK_LENGTH = 15
MIN_BLOCK_LENGTH = 10
MAX_BLOCK_LENGTH = 20
DEFAULT_N_RESAMPLES = 1999
DEFAULT_ALPHA = 0.05
DEFAULT_RANDOM_SEED = 0
BLOCK_START_OFFSET = 1
PVALUE_OFFSET = 1
SENSITIVITY_KIND = "non_overlapping_horizon_subsample"


@dataclass(frozen=True, slots=True)
class NonOverlapSensitivityResult:
    """Footnote bootstrap on a horizon-strided non-overlapping subsample.

    Attributes
    ----------
    model : str
        Challenger model name.
    horizon_days : int
        Stride used to thin the OOS differential.
    n_obs_full : int
        Length of the full aligned differential.
    n_obs_thinned : int
        Length after non-overlapping thinning.
    bootstrap : BootstrapResult or None
        Block-bootstrap result on the thinned series, if feasible.
    status : str
        ``ok`` or ``skipped_too_short``.

    Methods
    -------
    as_dict()
        Return JSON-friendly fields.
    describe()
        Return a short human-readable summary.
    """

    model: str
    horizon_days: int
    n_obs_full: int
    n_obs_thinned: int
    bootstrap: BootstrapResult | None
    status: str

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly fields.

        Returns
        -------
        dict of str to object
            Footnote sensitivity payload for one challenger.
        """
        payload: dict[str, object] = {
            "kind": SENSITIVITY_KIND,
            "model": self.model,
            "horizon_days": self.horizon_days,
            "n_obs_full": self.n_obs_full,
            "n_obs_thinned": self.n_obs_thinned,
            "status": self.status,
            "mean_delta_qlike": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "bootstrap_pvalue": None,
        }
        if self.bootstrap is not None:
            payload.update(
                {
                    "mean_delta_qlike": self.bootstrap.mean_delta,
                    "bootstrap_ci_low": self.bootstrap.ci_low,
                    "bootstrap_ci_high": self.bootstrap.ci_high,
                    "bootstrap_pvalue": self.bootstrap.pvalue,
                }
            )
        return payload

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact sensitivity summary.
        """
        return (
            f"{self.model}: status={self.status}, "
            f"n_thinned={self.n_obs_thinned}, horizon={self.horizon_days}"
        )


@dataclass(frozen=True, slots=True)
class BootstrapInferenceOptions:
    """Options for block-bootstrap inference on mean loss differentials.

    Parameters
    ----------
    block_length : int, default 15
        Contiguous block length in trading days (must be in 10–20).
    n_resamples : int, default 1999
        Number of bootstrap replications.
    alpha : float, default 0.05
        Two-sided significance level for the percentile CI / test.
    random_seed : int, default 0
        RNG seed for reproducible block draws.

    Methods
    -------
    validate()
        Raise if options are invalid.
    describe()
        Return a short human-readable summary.
    """

    block_length: int = DEFAULT_BLOCK_LENGTH
    n_resamples: int = DEFAULT_N_RESAMPLES
    alpha: float = DEFAULT_ALPHA
    random_seed: int = DEFAULT_RANDOM_SEED

    def validate(self) -> None:
        """Raise ``DataValidationError`` when options are invalid."""
        if self.block_length < MIN_BLOCK_LENGTH or self.block_length > MAX_BLOCK_LENGTH:
            raise DataValidationError(
                f"block_length must be in [{MIN_BLOCK_LENGTH}, {MAX_BLOCK_LENGTH}]."
            )
        if self.n_resamples < 1:
            raise DataValidationError("n_resamples must be at least 1.")
        if not 0.0 < self.alpha < 1.0:
            raise DataValidationError("alpha must be in (0, 1).")

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact description of bootstrap options.
        """
        return (
            f"block_length={self.block_length}, n_resamples={self.n_resamples}, "
            f"alpha={self.alpha}, random_seed={self.random_seed}"
        )


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Block-bootstrap inference for a mean loss differential.

    Attributes
    ----------
    mean_delta : float
        Sample mean of ``d_t``.
    ci_low : float
        Lower percentile confidence bound.
    ci_high : float
        Upper percentile confidence bound.
    pvalue : float
        Two-sided bootstrap p-value for H0: E[d] = 0.
    n_obs : int
        Length of the differential series.
    n_resamples : int
        Number of bootstrap replications used.

    Methods
    -------
    rejects_null(alpha)
        Return True when ``pvalue <= alpha``.
    as_dict()
        Return JSON-friendly fields.
    """

    mean_delta: float
    ci_low: float
    ci_high: float
    pvalue: float
    n_obs: int
    n_resamples: int

    def rejects_null(self, alpha: float = DEFAULT_ALPHA) -> bool:
        """Return whether the two-sided bootstrap test rejects H0.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
            True when ``pvalue <= alpha``.
        """
        return self.pvalue <= alpha

    def as_dict(self) -> dict[str, float | int]:
        """Return JSON-friendly fields.

        Returns
        -------
        dict of str to float or int
            Mean gap, CI bounds, p-value, and sample sizes.
        """
        return {
            "mean_delta": self.mean_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "pvalue": self.pvalue,
            "n_obs": self.n_obs,
            "n_resamples": self.n_resamples,
        }


@dataclass(frozen=True, slots=True)
class DMResult:
    """Diebold–Mariano result with HLN finite-sample correction.

    Attributes
    ----------
    dm_stat : float
        Uncorrected DM statistic.
    hln_stat : float
        HLN-corrected statistic.
    hln_pvalue : float
        Two-sided Student-t p-value from ``hln_stat``.
    nw_lags : int
        Newey–West lag used for HAC variance.
    n_obs : int
        Length of the differential series.
    horizon_days : int
        Forecast horizon used in the HLN correction.

    Methods
    -------
    rejects_null(alpha)
        Return True when ``hln_pvalue <= alpha``.
    as_dict()
        Return JSON-friendly fields.
    """

    dm_stat: float
    hln_stat: float
    hln_pvalue: float
    nw_lags: int
    n_obs: int
    horizon_days: int

    def rejects_null(self, alpha: float = DEFAULT_ALPHA) -> bool:
        """Return whether the HLN–DM test rejects H0.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
            True when ``hln_pvalue <= alpha``.
        """
        return self.hln_pvalue <= alpha

    def as_dict(self) -> dict[str, float | int]:
        """Return JSON-friendly fields.

        Returns
        -------
        dict of str to float or int
            DM / HLN statistics, p-value, and design metadata.
        """
        return {
            "dm_stat": self.dm_stat,
            "hln_stat": self.hln_stat,
            "hln_pvalue": self.hln_pvalue,
            "nw_lags": self.nw_lags,
            "n_obs": self.n_obs,
            "horizon_days": self.horizon_days,
        }


def nw_lags_for_horizon(horizon_days: int) -> int:
    """Return Newey–West lags locked to ``horizon_days - 1``.

    Parameters
    ----------
    horizon_days : int
        Forecast horizon in trading days.

    Returns
    -------
    int
        HAC lag count.

    Raises
    ------
    DataValidationError
        If ``horizon_days`` is less than 1.
    """
    if horizon_days < 1:
        raise DataValidationError("horizon_days must be at least 1.")
    return horizon_days - 1


def loss_differential(
        challenger_losses: pd.Series,
        baseline_losses: pd.Series,
) -> pd.Series:
    """Compute aligned per-row loss differentials.

    Parameters
    ----------
    challenger_losses : pandas.Series
        Per-row losses for the challenger model.
    baseline_losses : pandas.Series
        Per-row losses for the baseline model.

    Returns
    -------
    pandas.Series
        ``d_t = L_challenger - L_baseline`` on the inner-joined index.
        Negative mean favors the challenger under QLIKE.

    Raises
    ------
    DataValidationError
        If no overlapping finite observations remain.
    """
    aligned = pd.concat(
        [
            challenger_losses.rename("challenger"),
            baseline_losses.rename("baseline"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise DataValidationError(
            "No overlapping finite observations for loss differential."
        )
    differential = aligned["challenger"] - aligned["baseline"]
    differential.name = "loss_differential"
    return differential


def block_bootstrap_mean(
        differential: pd.Series,
        options: BootstrapInferenceOptions | None = None,
) -> BootstrapResult:
    """Estimate mean(d) uncertainty via a moving block bootstrap.

    Procedure
    ---------
    1. ``mu_hat = mean(d)``.
    2. Draw contiguous blocks of length ``block_length`` with replacement
       until length ``T``, truncate, and store bootstrap means ``mu_star``.
    3. Percentile CI from empirical ``alpha/2`` and ``1 - alpha/2`` quantiles.
    4. Two-sided p-value uses recentering: compare ``|mu_hat|`` to the
       distribution of ``|mu_star - mu_hat|``
       (Davidson–MacKinnon style count with +1 smoothing).

    Parameters
    ----------
    differential : pandas.Series
        Time-ordered loss differentials ``d_t``.
    options : BootstrapInferenceOptions or None
        Bootstrap settings; defaults are research-locked values.

    Returns
    -------
    BootstrapResult
        Mean gap, CI, and bootstrap p-value.

    Raises
    ------
    DataValidationError
        If ``differential`` is empty, too short for the block length,
        or options are invalid.
    """
    resolved = options if options is not None else BootstrapInferenceOptions()
    resolved.validate()
    values = _validated_differential_values(
        differential,
        resolved.block_length
    )
    mu_hat = float(np.mean(values))
    bootstrap_means = _collect_block_bootstrap_means(values, resolved)
    return _result_from_bootstrap_means(
        bootstrap_means,
        mu_hat,
        values.shape[0],
        resolved
    )


def _collect_block_bootstrap_means(
        values: np.ndarray,
        options: BootstrapInferenceOptions,
) -> np.ndarray:
    """Draw block-bootstrap replicates and return their means.

    Parameters
    ----------
    values : numpy.ndarray
        Time-ordered loss differentials.
    options : BootstrapInferenceOptions
        Validated bootstrap settings.

    Returns
    -------
    numpy.ndarray
        Bootstrap means of length ``options.n_resamples``.
    """
    rng = np.random.default_rng(options.random_seed)
    bootstrap_means = np.empty(options.n_resamples, dtype=float)
    for resample_index in range(options.n_resamples):
        sample = _draw_moving_block_sample(values, options.block_length, rng)
        bootstrap_means[resample_index] = float(np.mean(sample))
    return bootstrap_means


def _result_from_bootstrap_means(
        bootstrap_means: np.ndarray,
        mu_hat: float,
        n_obs: int,
        options: BootstrapInferenceOptions,
) -> BootstrapResult:
    """Build a ``BootstrapResult`` from bootstrap means and ``mu_hat``.

    Parameters
    ----------
    bootstrap_means : numpy.ndarray
        Means from each bootstrap replicate.
    mu_hat : float
        Observed mean of ``d_t``.
    n_obs : int
        Length of the original differential series.
    options : BootstrapInferenceOptions
        Validated bootstrap settings (alpha / n_resamples).

    Returns
    -------
    BootstrapResult
        Percentile CI and recentered two-sided p-value.
    """
    lower_q = options.alpha / 2.0
    ci_low = float(np.quantile(bootstrap_means, lower_q))
    ci_high = float(np.quantile(bootstrap_means, 1.0 - lower_q))
    exceedances = int(np.sum(np.abs(bootstrap_means - mu_hat) >= abs(mu_hat)))
    pvalue = (PVALUE_OFFSET + exceedances) / (PVALUE_OFFSET + options.n_resamples)
    return BootstrapResult(
        mean_delta=mu_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        pvalue=float(pvalue),
        n_obs=n_obs,
        n_resamples=options.n_resamples,
    )


def hln_diebold_mariano(
        differential: pd.Series,
        nw_lags: int,
        horizon_days: int | None = None,
) -> DMResult:
    """Diebold–Mariano test with Newey–West HAC and HLN correction.

    Parameters
    ----------
    differential : pandas.Series
        Time-ordered loss differentials ``d_t``.
    nw_lags : int
        Newey–West lag count (use ``nw_lags_for_horizon``).
    horizon_days : int or None, default None
        Forecast horizon for the HLN factor. When omitted, inferred as
        ``nw_lags + 1``.

    Returns
    -------
    DMResult
        Uncorrected DM, HLN-corrected statistic, and two-sided p-value.

    Raises
    ------
    DataValidationError
        If inputs are empty/invalid or HAC variance is non-positive.
    """
    if nw_lags < 0:
        raise DataValidationError("nw_lags must be non-negative.")
    resolved_horizon = nw_lags + 1 if horizon_days is None else horizon_days
    if resolved_horizon < 1:
        raise DataValidationError("horizon_days must be at least 1.")

    values = differential.dropna().to_numpy(dtype=float)
    n_obs = values.shape[0]
    if n_obs < 2:
        raise DataValidationError(
            "Need at least 2 observations for Diebold–Mariano."
        )

    mean_delta = float(np.mean(values))
    hac_variance = _newey_west_variance(values, nw_lags)
    if hac_variance <= 0.0:
        raise DataValidationError("Newey–West variance must be positive.")

    dm_stat = mean_delta / np.sqrt(hac_variance / n_obs)
    hln_factor = _hln_correction_factor(n_obs, resolved_horizon)
    hln_stat = float(dm_stat * hln_factor)
    degrees = n_obs - 1
    hln_pvalue = float(2.0 * stats.t.sf(abs(hln_stat), df=degrees))
    return DMResult(
        dm_stat=float(dm_stat),
        hln_stat=hln_stat,
        hln_pvalue=hln_pvalue,
        nw_lags=nw_lags,
        n_obs=n_obs,
        horizon_days=resolved_horizon,
    )


def _validated_differential_values(
        differential: pd.Series,
        block_length: int,
) -> np.ndarray:
    """Validate and return a float array for bootstrap."""
    values = differential.dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise DataValidationError("differential must be non-empty.")
    if values.size < block_length:
        raise DataValidationError(
            "differential length must be at least block_length."
        )
    return values


def _draw_moving_block_sample(
        values: np.ndarray,
        block_length: int,
        rng: np.random.Generator,
) -> np.ndarray:
    """Draw one moving-block bootstrap sample of length ``T``."""
    n_obs = values.shape[0]
    n_blocks = int(np.ceil(n_obs / block_length))
    max_start = n_obs - block_length
    starts = rng.integers(0, max_start + BLOCK_START_OFFSET, size=n_blocks)
    pieces = [values[start : start + block_length] for start in starts]
    sample = np.concatenate(pieces)[:n_obs]
    return sample


def _newey_west_variance(values: np.ndarray, nw_lags: int) -> float:
    """Estimate Newey–West HAC variance of the observations.

    Parameters
    ----------
    values : numpy.ndarray
        Loss differential series.
    nw_lags : int
        Maximum lag with Bartlett weights.

    Returns
    -------
    float
        HAC variance estimate of ``d_t`` (not of the mean).
    """
    n_obs = values.shape[0]
    demeaned = values - np.mean(values)
    gamma_zero = float(np.dot(demeaned, demeaned) / n_obs)
    variance = gamma_zero
    for lag in range(1, nw_lags + 1):
        weight = 1.0 - lag / (nw_lags + 1)
        gamma_lag = float(
            np.dot(demeaned[lag:], demeaned[:-lag]) / n_obs
        )
        variance += 2.0 * weight * gamma_lag
    return float(variance)


def _hln_correction_factor(n_obs: int, horizon_days: int) -> float:
    """Harvey–Leybourne–Newbold finite-sample correction factor.

    Parameters
    ----------
    n_obs : int
        Sample size ``T``.
    horizon_days : int
        Forecast horizon ``h``.

    Returns
    -------
    float
        Multiplicative HLN factor applied to the DM statistic.
    """
    numerator = (
        n_obs
        + 1
        - 2 * horizon_days
        + (horizon_days * (horizon_days - 1)) / n_obs
    )
    return float(np.sqrt(numerator / n_obs))


def non_overlapping_index(index: pd.Index, horizon_days: int) -> pd.Index:
    """Keep every ``horizon_days``-th label from a sorted unique index.

    Parameters
    ----------
    index : pandas.Index
        Session dates (or any ordered OOS index).
    horizon_days : int
        Stride matching the forecast horizon (e.g. 5 for ``target_rv_cc_5d``).

    Returns
    -------
    pandas.Index
        Thinned index (positional stride on sorted unique labels).

    Raises
    ------
    DataValidationError
        If ``horizon_days`` is less than 1 or ``index`` is empty.
    """
    if horizon_days < 1:
        raise DataValidationError("horizon_days must be at least 1.")
    ordered = pd.Index(index).sort_values().unique()
    if ordered.empty:
        raise DataValidationError("index must be non-empty.")
    return ordered[::horizon_days]


def non_overlapping_subsample(
    series: pd.Series,
    horizon_days: int,
) -> pd.Series:
    """Thin a time-ordered series to non-overlapping horizon spacing.

    Parameters
    ----------
    series : pandas.Series
        Loss differentials (or losses) indexed by session date.
    horizon_days : int
        Keep every ``horizon_days``-th observation after sorting.

    Returns
    -------
    pandas.Series
        Copy of the thinned series.

    Raises
    ------
    DataValidationError
        If the series is empty after dropna, or ``horizon_days`` is invalid.
    """
    cleaned = series.dropna().sort_index()
    if cleaned.empty:
        raise DataValidationError("series must be non-empty after dropna.")
    kept = non_overlapping_index(cleaned.index, horizon_days)
    return cleaned.loc[kept].copy()


def block_bootstrap_nonoverlap_sensitivity(
    differential: pd.Series,
    horizon_days: int,
    options: BootstrapInferenceOptions | None = None,
) -> tuple[BootstrapResult | None, int, int, str]:
    """Block-bootstrap mean(d) on a non-overlapping horizon subsample.

    This is a footnote sensitivity check. It does not replace the primary
    bootstrap on the full overlapping OOS series.

    Parameters
    ----------
    differential : pandas.Series
        Full aligned ``d_t`` series.
    horizon_days : int
        Thinning stride (locked research default: 5).
    options : BootstrapInferenceOptions or None
        Same bootstrap defaults as the primary test.

    Returns
    -------
    bootstrap : BootstrapResult or None
        Result when the thinned series is long enough for ``block_length``.
    n_obs_full : int
        Length of the full differential.
    n_obs_thinned : int
        Length after thinning.
    status : str
        ``ok`` or ``skipped_too_short``.

    Raises
    ------
    DataValidationError
        If inputs are empty/invalid (other than thinned-too-short).
    """
    resolved = options if options is not None else BootstrapInferenceOptions()
    resolved.validate()
    full = differential.dropna().sort_index()
    n_full = int(full.shape[0])
    thinned = non_overlapping_subsample(full, horizon_days)
    n_thinned = int(thinned.shape[0])
    if n_thinned < resolved.block_length:
        return None, n_full, n_thinned, "skipped_too_short"
    result = block_bootstrap_mean(thinned, resolved)
    return result, n_full, n_thinned, "ok"
