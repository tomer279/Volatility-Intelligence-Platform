"""Helpers for summarizing walk-forward metric tables.

Exports
-------
InferenceSummaryOptions
    Baseline, bootstrap, HLN–DM, and horizon settings for enrichment.
summarize_walk_forward
    Aggregate fold metrics by model and sort by a primary metric.
summarize_with_inference
    Horse-race means plus bootstrap (and optional HLN–DM) vs baseline.
summarize_nonoverlap_sensitivity
    Footnote bootstrap table on every-horizon-day OOS differentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.evaluation.inference import (
    BootstrapInferenceOptions,
    BootstrapResult,
    DMResult,
    block_bootstrap_mean,
    hln_diebold_mariano,
    loss_differential,
    nw_lags_for_horizon,
)
from vip.evaluation.inference import (
    NonOverlapSensitivityResult,
    block_bootstrap_nonoverlap_sensitivity,
)

PRIMARY_METRIC = "qlike"
DEFAULT_BASELINE_MODEL = "har_rv_ols"
DEFAULT_HORIZON_DAYS = 5
METRIC_COLUMNS: tuple[str, ...] = ("qlike", "mse", "mae")
INFERENCE_COLUMNS: tuple[str, ...] = (
    "mean_delta_qlike",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_pvalue",
    "significant_vs_baseline",
    "dm_stat",
    "hln_stat",
    "hln_pvalue",
    "nw_lags",
)
OOS_LOSS_REQUIRED: tuple[str, ...] = ("model", "qlike_loss")


@dataclass(frozen=True, slots=True)
class InferenceSummaryOptions:
    """Settings for inference-enriched horse-race summaries.

    Parameters
    ----------
    baseline_model : str, default ``'har_rv_ols'``
        Reference model for loss differentials.
    horizon_days : int, default 5
        Forecast horizon; NW lags = ``horizon_days - 1``.
    include_hln_dm : bool, default True
        When True, attach secondary HLN–DM columns.
    bootstrap : BootstrapInferenceOptions
        Block-bootstrap settings (primary inference).

    Methods
    -------
    validate()
        Raise if settings are invalid.
    describe()
        Return a short human-readable summary.
    """

    baseline_model: str = DEFAULT_BASELINE_MODEL
    horizon_days: int = DEFAULT_HORIZON_DAYS
    include_hln_dm: bool = True
    bootstrap: BootstrapInferenceOptions = field(
        default_factory=BootstrapInferenceOptions
    )

    def validate(self) -> None:
        """Raise ``DataValidationError`` when settings are invalid."""
        if not self.baseline_model.strip():
            raise DataValidationError("baseline_model must be non-empty.")
        if self.horizon_days < 1:
            raise DataValidationError("horizon_days must be at least 1.")
        self.bootstrap.validate()

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact description of inference summary options.
        """
        return (
            f"baseline={self.baseline_model}, horizon_days={self.horizon_days}, "
            f"include_hln_dm={self.include_hln_dm}, {self.bootstrap.describe()}"
        )


def summarize_walk_forward(
        fold_metrics: pd.DataFrame,
        primary_metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Aggregate walk-forward fold metrics by model.

    Parameters
    ----------
    fold_metrics : pandas.DataFrame
        Output of ``run_walk_forward``.
    primary_metric : str, default ``'qlike'``
        Metric used for ascending sort (lower is better).

    Returns
    -------
    pandas.DataFrame
        One row per model with mean metric values, sorted by
        ``primary_metric`` ascending.

    Raises
    ------
    DataValidationError
        If required columns are missing or ``primary_metric`` is unknown.
    """
    required = {"model", *METRIC_COLUMNS}
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"fold_metrics missing required columns: {missing_text}."
        )
    if primary_metric not in METRIC_COLUMNS:
        raise DataValidationError(
            f"Unsupported primary metric '{primary_metric}'."
        )

    summary = (
        fold_metrics.groupby("model", sort=False)[list(METRIC_COLUMNS)]
        .mean()
        .reset_index()
        .sort_values(primary_metric, ascending=True)
        .reset_index(drop=True)
    )
    return summary


def summarize_with_inference(
        fold_metrics: pd.DataFrame,
        oos_losses: pd.DataFrame,
        options: InferenceSummaryOptions | None = None,
) -> pd.DataFrame:
    """Build a horse-race table with primary bootstrap inference columns.

    Mean QLIKE / MSE / MAE come from fold aggregates (descriptive).
    Inference columns use per-row OOS QLIKE differentials vs the baseline.
    The baseline row keeps metrics only; Δ / p columns are null.

    Parameters
    ----------
    fold_metrics : pandas.DataFrame
        Output of ``run_walk_forward``.
    oos_losses : pandas.DataFrame
        Output of ``collect_walk_forward_oos_losses`` (date index;
        columns include ``model``, ``qlike_loss``).
    options : InferenceSummaryOptions or None
        Baseline / bootstrap / HLN settings.

    Returns
    -------
    pandas.DataFrame
        Sorted horse-race rows with locked inference columns.

    Raises
    ------
    DataValidationError
        If inputs are missing columns, baseline is absent, or options
        are invalid.
    """
    resolved = options if options is not None else InferenceSummaryOptions()
    resolved.validate()
    _validate_oos_losses(oos_losses, resolved.baseline_model)

    summary = summarize_walk_forward(fold_metrics, primary_metric=PRIMARY_METRIC)
    baseline_losses = _model_loss_series(oos_losses, resolved.baseline_model)
    enriched_rows = [
        _enrich_summary_row(row, baseline_losses, oos_losses, resolved)
        for row in summary.to_dict(orient="records")
    ]
    return pd.DataFrame(enriched_rows)


def _validate_oos_losses(oos_losses: pd.DataFrame, baseline_model: str) -> None:
    """Validate the OOS loss panel for inference enrichment."""
    missing = sorted(set(OOS_LOSS_REQUIRED).difference(oos_losses.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"oos_losses missing required columns: {missing_text}."
        )
    if oos_losses.empty:
        raise DataValidationError("oos_losses must be non-empty.")
    if baseline_model not in set(oos_losses["model"].astype(str)):
        raise DataValidationError(
            f"Baseline model '{baseline_model}' not found in oos_losses."
        )


def _model_loss_series(oos_losses: pd.DataFrame, model_name: str) -> pd.Series:
    """Return date-indexed QLIKE losses for one model (copy)."""
    mask = oos_losses["model"].astype(str) == model_name
    subset = oos_losses.loc[mask, "qlike_loss"]
    return subset.sort_index().copy()


def _enrich_summary_row(
        row: dict[str, object],
        baseline_losses: pd.Series,
        oos_losses: pd.DataFrame,
        options: InferenceSummaryOptions,
) -> dict[str, object]:
    """Attach inference fields to one horse-race summary row."""
    model_name = str(row["model"])
    blank = _blank_inference_fields(options.include_hln_dm)
    if model_name == options.baseline_model:
        return {**row, **blank}

    challenger_losses = _model_loss_series(oos_losses, model_name)
    differential = loss_differential(challenger_losses, baseline_losses)
    bootstrap = block_bootstrap_mean(differential, options.bootstrap)
    hln = _optional_hln_result(differential, options)
    return {
        **row,
        **_inference_fields_from_results(bootstrap, hln, options),
    }


def _optional_hln_result(
        differential: pd.Series,
        options: InferenceSummaryOptions,
) -> DMResult | None:
    """Run HLN–DM when enabled; otherwise return None."""
    if not options.include_hln_dm:
        return None
    nw_lags = nw_lags_for_horizon(options.horizon_days)
    return hln_diebold_mariano(
        differential,
        nw_lags=nw_lags,
        horizon_days=options.horizon_days,
    )


def _inference_fields_from_results(
        bootstrap: BootstrapResult,
        hln: DMResult | None,
        options: InferenceSummaryOptions,
) -> dict[str, object]:
    """Map bootstrap / HLN results onto locked column names."""
    significant = bool(
        bootstrap.rejects_null(options.bootstrap.alpha)
        and bootstrap.mean_delta < 0.0
    )
    fields: dict[str, object] = {
        "mean_delta_qlike": bootstrap.mean_delta,
        "bootstrap_ci_low": bootstrap.ci_low,
        "bootstrap_ci_high": bootstrap.ci_high,
        "bootstrap_pvalue": bootstrap.pvalue,
        "significant_vs_baseline": significant,
        "dm_stat": None,
        "hln_stat": None,
        "hln_pvalue": None,
        "nw_lags": None,
    }
    if hln is not None:
        fields["dm_stat"] = hln.dm_stat
        fields["hln_stat"] = hln.hln_stat
        fields["hln_pvalue"] = hln.hln_pvalue
        fields["nw_lags"] = hln.nw_lags
    return fields


def _blank_inference_fields(include_hln_dm: bool) -> dict[str, object]:
    """Null inference columns for the baseline row."""
    fields: dict[str, object] = {
        "mean_delta_qlike": None,
        "bootstrap_ci_low": None,
        "bootstrap_ci_high": None,
        "bootstrap_pvalue": None,
        "significant_vs_baseline": None,
        "dm_stat": None,
        "hln_stat": None,
        "hln_pvalue": None,
        "nw_lags": None,
    }
    if not include_hln_dm:
        # Keep keys present so the schema stays stable for artifacts/HTML.
        pass
    return fields


def summarize_nonoverlap_sensitivity(
        oos_losses: pd.DataFrame,
        options: InferenceSummaryOptions | None = None,
) -> pd.DataFrame:
    """Footnote bootstrap on every-``horizon_days``-th OOS differential.

    Reuses the same baseline alignment as ``summarize_with_inference``.
    Does not alter primary horse-race significance columns.

    Parameters
    ----------
    oos_losses : pandas.DataFrame
        Per-row OOS loss panel (``model``, ``qlike_loss``).
    options : InferenceSummaryOptions or None
        Baseline / horizon / bootstrap settings.

    Returns
    -------
    pandas.DataFrame
        One row per non-baseline model (JSON-friendly columns).

    Raises
    ------
    DataValidationError
        If ``oos_losses`` or options are invalid.
    """
    resolved = options if options is not None else InferenceSummaryOptions()
    resolved.validate()
    _validate_oos_losses(oos_losses, resolved.baseline_model)
    baseline = _model_loss_series(oos_losses, resolved.baseline_model)
    models = sorted(
        {
            str(name)
            for name in oos_losses["model"].astype(str).unique()
            if str(name) != resolved.baseline_model
        }
    )
    rows = [
        _nonoverlap_row_for_model(model_name, baseline, oos_losses, resolved)
        for model_name in models
    ]
    return pd.DataFrame(rows)


def _nonoverlap_row_for_model(
        model_name: str,
        baseline_losses: pd.Series,
        oos_losses: pd.DataFrame,
        options: InferenceSummaryOptions,
) -> dict[str, object]:
    """Build one non-overlap sensitivity record for a challenger."""
    challenger = _model_loss_series(oos_losses, model_name)
    differential = loss_differential(challenger, baseline_losses)
    bootstrap, n_full, n_thinned, status = block_bootstrap_nonoverlap_sensitivity(
        differential,
        options.horizon_days,
        options.bootstrap,
    )
    result = NonOverlapSensitivityResult(
        model=model_name,
        horizon_days=options.horizon_days,
        n_obs_full=n_full,
        n_obs_thinned=n_thinned,
        bootstrap=bootstrap,
        status=status,
    )
    return result.as_dict()
