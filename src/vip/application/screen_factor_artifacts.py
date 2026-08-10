"""Persist factor-screen JSON, plots, and HTML artifacts.

Exports
-------
persist_screen_artifacts
    Write metrics, inference, importance, plots, and report.html for one run.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

import pandas as pd
from vip.domain.value_objects import ExperimentId
from vip.evaluation.comparison import summarize_nonoverlap_sensitivity
from vip.evaluation.inference import nw_lags_for_horizon
from vip.features.targets import TARGET_NAME_PREFIX
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.reporting.experiment_summary import (
    InferenceReportMeta,
    ReportMeta,
    ScreenReportPayload,
    build_factor_screen_context,
)
from vip.reporting.html_report import render_factor_screen_report, write_html_report
from vip.visualization.importance_plots import (
    ImportancePlotOptions,
    plot_importance_bars,
)

if TYPE_CHECKING:
    from vip.application.screen_factors import (
        FactorScreenResult,
        ScreenArtifactContext,
        ScreenResultTables,
        ShapScreenOutputs,
    )


_INFERENCE_RECORD_COLUMNS: tuple[str, ...] = (
    "model",
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


def persist_screen_artifacts(
        artifact_store: FilesystemArtifactStore,
        result: FactorScreenResult,
        context: ScreenArtifactContext,
) -> None:
    """Write screen artifacts (JSON, plots, HTML report).

    Parameters
    ----------
    artifact_store : FilesystemArtifactStore
        Destination store for the experiment directory.
    result : FactorScreenResult
        Completed screen tables and identity.
    context : ScreenArtifactContext
        Validated config + inference used for meta / report fields.
    """
    _write_screen_json_artifacts(artifact_store, result, context)
    _write_screen_plots(
        artifact_store,
        result.identity.experiment_id,
        result.tables,
        result.shap,
    )
    _write_screen_html_report(artifact_store, result, context)


def _write_screen_json_artifacts(
        artifact_store: FilesystemArtifactStore,
        result: FactorScreenResult,
        context: ScreenArtifactContext,
) -> None:
    """Write JSON artifacts for a factor-screen run."""
    tables = result.tables
    identity = result.identity
    experiment_id = identity.experiment_id
    inference = context.inference

    artifact_store.write_json(
        experiment_id, "metrics", _records_with_nulls(tables.summary)
    )
    artifact_store.write_json(
        experiment_id,
        "folds",
        tables.fold_metrics.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "importance",
        tables.importance.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "factor_ranking",
        tables.ranking.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "metrics_by_regime",
        tables.regime_metrics.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id, "oos_losses", _oos_losses_records(tables.oos_losses)
    )
    artifact_store.write_json(
        experiment_id,
        "inference",
        _inference_records(tables.summary, inference.baseline_model),
    )
    if inference.include_nonoverlap_sensitivity:
        sensitivity = summarize_nonoverlap_sensitivity(
            tables.oos_losses,
            inference.to_summary_options(),
        )
        artifact_store.write_json(
            experiment_id,
            "inference_sensitivity",
            _records_with_nulls(sensitivity),
        )
    artifact_store.write_json(
        experiment_id,
        "screen_meta",
        context.screen_meta_payload(identity, result.top_feature()),
    )


def _write_screen_plots(
        artifact_store: FilesystemArtifactStore,
        experiment_id: ExperimentId,
        tables: ScreenResultTables,
        shap_outputs: ShapScreenOutputs | None,
) -> Path:
    """Write plot PNGs and return permutation importance plot path."""
    plot_path = artifact_store.experiment_dir(experiment_id) / "importance_plot.png"
    plot_importance_bars(tables.ranking, plot_path)

    if shap_outputs is not None:
        shap_plot_path = (
            artifact_store.experiment_dir(experiment_id) / "shap_importance_plot.png"
        )
        plot_importance_bars(
            shap_outputs.ranking,
            shap_plot_path,
            options=ImportancePlotOptions(title="TreeSHAP importance (median/mean |SHAP|)"),
        )
        artifact_store.write_json(
            experiment_id,
            "shap_importance",
            shap_outputs.importance.to_dict(orient="records"),
        )
        artifact_store.write_json(
            experiment_id,
            "shap_ranking",
            shap_outputs.ranking.to_dict(orient="records"),
        )

    return plot_path


def _write_screen_html_report(
        artifact_store: FilesystemArtifactStore,
        result: FactorScreenResult,
        context: ScreenArtifactContext,
) -> None:
    """Build and write the factor-screen HTML report."""
    identity = result.identity
    tables = result.tables
    config = context.config
    inference = context.inference
    experiment_id = identity.experiment_id
    plot_path = artifact_store.experiment_dir(experiment_id) / "importance_plot.png"

    payload = ScreenReportPayload(
        symbol=identity.symbol.value,
        experiment_id=experiment_id.value,
        screening_model=identity.screening_model,
        summary=tables.summary,
        ranking=tables.ranking,
        regime_metrics=tables.regime_metrics,
    )
    meta = ReportMeta(
        target_column=_target_column_for_horizon(inference.horizon_days),
        n_splits=config.n_splits,
        embargo_size=config.embargo_size,
        inference=InferenceReportMeta(
            baseline_model=inference.baseline_model,
            nw_lags=nw_lags_for_horizon(inference.horizon_days),
            bootstrap_block_length=inference.bootstrap.block_length,
            alpha=inference.bootstrap.alpha,
            bootstrap_n_resamples=inference.bootstrap.n_resamples,
        ),
    )
    report_context = build_factor_screen_context(payload, plot_path, meta)
    html = render_factor_screen_report(report_context)
    write_html_report(
        artifact_store.experiment_dir(experiment_id) / "report.html",
        html,
    )


def _oos_losses_records(oos_losses: pd.DataFrame) -> list[dict[str, object]]:
    """JSON-friendly long-form OOS loss panel with a date column."""
    frame = oos_losses.reset_index(names="date").copy()
    frame["date"] = frame["date"].map(_stringify_timestamp)
    return _records_with_nulls(frame)


def _inference_records(summary: pd.DataFrame, baseline_model: str) -> list[dict]:
    """Challenger-only inference rows for inference.json."""
    challengers = summary.loc[summary["model"].astype(str) != baseline_model]
    present = [
        col for col in _INFERENCE_RECORD_COLUMNS if col in challengers.columns
    ]
    return _records_with_nulls(challengers[list(present)])


def _stringify_timestamp(value: object) -> str:
    """Convert a timestamp-like value to an ISO date string."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _records_with_nulls(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a frame to records, mapping NaN to None for JSON."""
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        cleaned = {
            key: (None if _is_null(value) else value)
            for key, value in row.items()
        }
        records.append(cleaned)
    return records


def _is_null(value: object) -> bool:
    """Return True for pandas/NumPy missing values."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _target_column_for_horizon(horizon_days: int) -> str:
    """Return ``target_rv_cc_{h}d`` (kept local to avoid importing screen_factors)."""
    return f"{TARGET_NAME_PREFIX}{horizon_days}d"
