"""Multi-horizon factor-screen orchestration.

Exports
-------
MultiHorizonStores
    Market, feature, and artifact-root dependencies for a study.
MultiHorizonScreenConfig
    Symbol, horizons, VIX / skip flags, and screen tuning.
MultiHorizonScreenResult
    Study id, summary table, and per-horizon experiment ids.
screen_multi_horizon
    Build/load features per horizon, call ``screen_factors``, stack summary.
MultiHorizonInferenceOverrides
    Optional bootstrap / HLN overrides applied on top of ``settings_for_horizon``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import pandas as pd

from vip.application.build_feature_matrix import (
    FeatureMatrixExtras,
    build_and_persist_feature_matrix,
)
from vip.application.screen_factors import (
    DEFAULT_BASELINE_MODEL,
    FactorScreenResult,
    ScreenConfig,
    screen_factors,
    settings_for_horizon,
    target_column_for_horizon,
    ScreenInferenceOptions
)
from vip.domain.errors import DataValidationError, PersistenceError
from vip.domain.value_objects import ExperimentId, Symbol
from vip.evaluation.horizon_defaults import LOCKED_SCREEN_HORIZONS
from vip.evaluation.inference import nw_lags_for_horizon
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.persistence.parquet_store import ParquetMarketDataStore
from vip.reporting.experiment_summary import (
    MultiHorizonReportMeta,
    MultiHorizonReportPayload,
    build_multi_horizon_context,
)
from vip.reporting.html_report import (
    render_multi_horizon_screen_report,
    write_html_report,
)

HORIZON_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon_days",
    "model",
    "qlike",
    "mse",
    "mae",
    "mean_delta_qlike",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_pvalue",
    "significant_vs_baseline",
)

_SUMMARY_METRIC_COLUMNS: tuple[str, ...] = (
    "model",
    "qlike",
    "mse",
    "mae",
    "mean_delta_qlike",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_pvalue",
    "significant_vs_baseline",
)


@dataclass(frozen=True, slots=True)
class MultiHorizonStores:
    """Persistence dependencies for a multi-horizon screen.

    Parameters
    ----------
    market_store : ParquetMarketDataStore
        Raw OHLCV store (used when features are built).
    feature_store : ParquetFeatureMatrixStore
        Processed feature-matrix store (one target per build).
    artifact_root : pathlib.Path
        Root artifacts directory (for example ``data/artifacts``).

    Methods
    -------
    describe()
        Return a short human-readable summary.
    resolve_study_dir(study_id)
        Return the study root directory under ``artifact_root``.
    """

    market_store: ParquetMarketDataStore
    feature_store: ParquetFeatureMatrixStore
    artifact_root: Path

    def describe(self) -> str:
        """Return a short human-readable summary."""
        return f"artifact_root={self.artifact_root}"

    def resolve_study_dir(self, study_id: ExperimentId) -> Path:
        """Return the study root directory under ``artifact_root``."""
        return self.artifact_root / study_id.as_path_key()


@dataclass(frozen=True, slots=True)
class MultiHorizonInferenceOverrides:
    """Optional inference overrides for each horizon screen.

    Applied on top of ``settings_for_horizon(h).inference``. Embargo and
    block length always come from the horizon defaults, not from here.

    Parameters
    ----------
    bootstrap_n_resamples : int or None
        When set, override bootstrap replication count (useful in tests).
    include_hln_dm : bool
        Forwarded into per-horizon ``ScreenInferenceOptions``.
    include_nonoverlap_sensitivity : bool
        Forwarded into per-horizon ``ScreenInferenceOptions``.

    Methods
    -------
    validate()
        Raise if overrides are invalid.
    describe()
        Return a short human-readable summary.
    """

    bootstrap_n_resamples: int | None = None
    include_hln_dm: bool = True
    include_nonoverlap_sensitivity: bool = True

    def validate(self) -> None:
        """Raise ``DataValidationError`` when overrides are invalid."""
        if self.bootstrap_n_resamples is not None and self.bootstrap_n_resamples < 1:
            raise DataValidationError("bootstrap_n_resamples must be >= 1.")

    def describe(self) -> str:
        """Return a short human-readable summary."""
        n_resamples = (
            "default"
            if self.bootstrap_n_resamples is None
            else str(self.bootstrap_n_resamples)
        )
        return (
            f"n_resamples={n_resamples}, "
            f"hln_dm={self.include_hln_dm}, "
            f"nonoverlap={self.include_nonoverlap_sensitivity}"
        )


@dataclass(frozen=True, slots=True)
class MultiHorizonScreenConfig:
    """Settings for a multi-horizon factor screen.

    Parameters
    ----------
    symbol : Symbol
        Instrument to screen.
    horizons : tuple of int
        Forecast horizons in trading days (default locked ``1, 5, 21``).
    with_vix : bool
        When True, join VIX features during feature builds.
    with_jump_features : bool
        When True, include the daily jump-robust feature family.
    skip_features : bool
        When True, require an existing matrix with the per-horizon target.
    screen_config : ScreenConfig
        Walk-forward / importance tuning. ``embargo_size`` is overwritten
        per horizon via ``settings_for_horizon``.
    inference : MultiHorizonInferenceOverrides
        Optional bootstrap / HLN overrides on horizon defaults.

    Methods
    -------
    validate()
        Raise if settings are invalid.
    describe()
        Return a short human-readable summary.
    """

    symbol: Symbol
    horizons: tuple[int, ...] = LOCKED_SCREEN_HORIZONS
    with_vix: bool = False
    skip_features: bool = False
    screen_config: ScreenConfig = field(default_factory=ScreenConfig)
    inference: MultiHorizonInferenceOverrides = field(
        default_factory=MultiHorizonInferenceOverrides
    )
    with_jump_features: bool = False

    def validate(self) -> None:
        """Raise ``DataValidationError`` when configuration is invalid."""
        if not self.horizons:
            raise DataValidationError("horizons must be a non-empty sequence.")
        for horizon in self.horizons:
            if horizon not in LOCKED_SCREEN_HORIZONS:
                raise DataValidationError(
                    f"horizon_days must be one of {LOCKED_SCREEN_HORIZONS}; "
                    f"got {horizon}."
                )
        self.inference.validate()
        self.screen_config.validate()

    def describe(self) -> str:
        """Return a short human-readable summary."""
        joined = ",".join(str(h) for h in self.horizons)
        return (
            f"symbol={self.symbol.value}, horizons=[{joined}], "
            f"with_vix={self.with_vix}, "
            f"with_jump_features={self.with_jump_features}, "
            f"skip_features={self.skip_features}"
        )


@dataclass(frozen=True, slots=True)
class MultiHorizonScreenResult:
    """Outputs from a multi-horizon factor screen.

    Parameters
    ----------
    study_id : ExperimentId
        Study namespace under the artifact root.
    summary : pandas.DataFrame
        Cross-horizon horse-race table (``HORIZON_SUMMARY_COLUMNS``).
    horizon_experiment_ids : dict of int to ExperimentId
        Per-horizon ids returned by ``screen_factors`` (pre-promote).

    Methods
    -------
    describe()
        Return a short human-readable summary.
    horizon_count()
        Return the number of horizons screened.
    """

    study_id: ExperimentId
    summary: pd.DataFrame
    horizon_experiment_ids: dict[int, ExperimentId]

    def describe(self) -> str:
        """Return a short human-readable summary."""
        return (
            f"{self.study_id.value} / horizons={self.horizon_count()} / "
            f"rows={len(self.summary)}"
        )

    def horizon_count(self) -> int:
        """Return the number of horizons screened."""
        return int(len(self.horizon_experiment_ids))


def screen_multi_horizon(
        stores: MultiHorizonStores,
        config: MultiHorizonScreenConfig,
) -> MultiHorizonScreenResult:
    """Run factor screens across horizons and write a cross-horizon study.

    For each horizon ``h``: optionally rebuild features with
    ``target_rv_cc_{h}d``, call ``screen_factors`` with
    ``settings_for_horizon(h)``, promote artifacts into ``h{h}d/``, then
    stack metrics into ``horizon_summary.json``.

    Parameters
    ----------
    stores : MultiHorizonStores
        Market, feature, and artifact-root dependencies.
    config : MultiHorizonScreenConfig
        Symbol, horizons, and tuning flags.

    Returns
    -------
    MultiHorizonScreenResult
        Study id, stacked summary, and per-horizon experiment ids.

    Raises
    ------
    DataValidationError
        If configuration is invalid.
    PersistenceError
        If required caches or targets are missing.
    """
    config.validate()
    study_id = _build_study_id(config.symbol)
    study_dir = stores.resolve_study_dir(study_id)
    study_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    horizon_ids: dict[int, ExperimentId] = {}
    per_horizon_meta: list[dict[str, object]] = []

    for horizon_days in config.horizons:
        _ensure_features_for_horizon(stores, config, horizon_days)
        defaults = settings_for_horizon(horizon_days)
        screen_cfg = replace(
            config.screen_config,
            embargo_size=defaults.config.embargo_size,
        )
        inference = _resolve_inference(defaults.inference, config)
        horizon_dir = study_dir / f"h{horizon_days}d"
        horizon_store = FilesystemArtifactStore(root_dir=horizon_dir)
        result = screen_factors(
            feature_store=stores.feature_store,
            artifact_store=horizon_store,
            symbol=config.symbol,
            config=screen_cfg,
            inference=inference,
        )
        _promote_experiment_artifacts(horizon_dir, result.identity.experiment_id)
        horizon_ids[horizon_days] = result.identity.experiment_id
        summary_rows.extend(_horizon_summary_rows(horizon_days, result))
        per_horizon_meta.append(_horizon_meta_row(horizon_days, inference, screen_cfg))

    summary = pd.DataFrame.from_records(summary_rows, columns=list(HORIZON_SUMMARY_COLUMNS))
    _write_study_artifacts(
        study_dir=study_dir,
        config=config,
        summary=summary,
        per_horizon_meta=per_horizon_meta,
        study_id=study_id,
    )
    return MultiHorizonScreenResult(
        study_id=study_id,
        summary=summary,
        horizon_experiment_ids=horizon_ids,
    )


def _build_study_id(symbol: Symbol) -> ExperimentId:
    """Build ``multi-horizon-screen-{symbol}-{date}``."""
    return ExperimentId(
        f"multi-horizon-screen-{symbol.as_path_key().lower()}-{date.today().isoformat()}"
    )


def _ensure_features_for_horizon(
        stores: MultiHorizonStores,
        config: MultiHorizonScreenConfig,
        horizon_days: int,
) -> None:
    """Build or validate the feature matrix for one horizon."""
    target = target_column_for_horizon(horizon_days)
    if config.skip_features:
        if not stores.feature_store.exists(config.symbol):
            raise PersistenceError(
                f"Missing feature matrix for {config.symbol.value}, "
                "but skip_features was set."
            )
        matrix = stores.feature_store.load(config.symbol)
        if target not in matrix.columns:
            raise PersistenceError(
                f"Feature matrix missing target column '{target}'."
            )
        return

    build_and_persist_feature_matrix(
        market_store=stores.market_store,
        feature_store=stores.feature_store,
        symbol=config.symbol,
        horizon_days=horizon_days,
        extras=FeatureMatrixExtras(
            include_vix=config.with_vix,
            include_jump=config.with_jump_features,
        ),
    )


def _resolve_inference(
        defaults_inference: ScreenInferenceOptions,
        config: MultiHorizonScreenConfig,
) -> ScreenInferenceOptions:
    """Apply optional overrides onto horizon defaults."""
    overrides = config.inference
    inference = replace(
        defaults_inference,
        include_hln_dm=overrides.include_hln_dm,
        include_nonoverlap_sensitivity=overrides.include_nonoverlap_sensitivity,
    )
    if overrides.bootstrap_n_resamples is None:
        return inference
    return replace(
        inference,
        bootstrap=replace(
            inference.bootstrap,
            n_resamples=overrides.bootstrap_n_resamples,
        ),
    )


def _promote_experiment_artifacts(
        horizon_dir: Path,
        experiment_id: ExperimentId,
) -> None:
    """Move ``h{h}d/{experiment_id}/*`` up to ``h{h}d/``."""
    nested = horizon_dir / experiment_id.as_path_key()
    if not nested.is_dir():
        raise PersistenceError(
            f"Expected screen artifacts under {nested}."
        )
    for child in nested.iterdir():
        destination = horizon_dir / child.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        child.rename(destination)
    nested.rmdir()


def _horizon_summary_rows(
        horizon_days: int,
        result: FactorScreenResult,
) -> list[dict[str, object]]:
    """Tag each horse-race row with ``horizon_days``."""
    frame = result.tables.summary.copy()
    missing = [c for c in _SUMMARY_METRIC_COLUMNS if c not in frame.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"screen summary missing columns: {missing_text}."
        )
    rows: list[dict[str, object]] = []
    for record in frame.loc[:, list(_SUMMARY_METRIC_COLUMNS)].to_dict(orient="records"):
        cleaned = {key: _null_if_missing(value) for key, value in record.items()}
        cleaned["horizon_days"] = horizon_days
        rows.append(cleaned)
    return rows


def _horizon_meta_row(horizon_days: int, inference, screen_cfg: ScreenConfig) -> dict[str, object]:
    """One per-horizon block for study-level ``screen_meta.json``."""
    return {
        "horizon_days": horizon_days,
        "target_column": target_column_for_horizon(horizon_days),
        "embargo_size": screen_cfg.embargo_size,
        "nw_lags": nw_lags_for_horizon(horizon_days),
        "bootstrap_block_length": inference.bootstrap.block_length,
        "bootstrap_n_resamples": inference.bootstrap.n_resamples,
        "alpha": inference.bootstrap.alpha,
        "baseline_model": inference.baseline_model,
    }


def _write_study_artifacts(
        study_dir: Path,
        config: MultiHorizonScreenConfig,
        summary: pd.DataFrame,
        per_horizon_meta: list[dict[str, object]],
        study_id: ExperimentId,
) -> None:
    """Persist study-level meta, summary JSON, and HTML memo."""
    alpha = float(per_horizon_meta[0]["alpha"]) if per_horizon_meta else 0.05
    baseline = (
        str(per_horizon_meta[0]["baseline_model"])
        if per_horizon_meta
        else DEFAULT_BASELINE_MODEL
    )
    meta_payload = {
        "symbol": config.symbol.value,
        "study_id": study_id.value,
        "horizons": list(config.horizons),
        "baseline_model": baseline,
        "alpha": alpha,
        "with_vix": config.with_vix,
        "with_jump_features": config.with_jump_features,
        "per_horizon": per_horizon_meta,
    }
    _write_json(study_dir / "screen_meta.json", meta_payload)
    _write_json(
        study_dir / "horizon_summary.json",
        _records_with_nulls(summary),
    )

    payload = MultiHorizonReportPayload(
        symbol=config.symbol.value,
        study_id=study_id.value,
        horizon_summary=summary,
    )
    report_meta = MultiHorizonReportMeta(
        horizons=config.horizons,
        baseline_model=baseline,
        alpha=alpha,
        per_horizon=tuple(per_horizon_meta),
    )
    context = build_multi_horizon_context(payload, report_meta)
    html = render_multi_horizon_screen_report(context)
    write_html_report(study_dir / "report.html", html)


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON artifact next to the study root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise PersistenceError(f"Failed to write {path}: {exc}") from exc


def _records_with_nulls(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a frame to records, mapping NaN to None."""
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        records.append({key: _null_if_missing(value) for key, value in row.items()})
    return records


def _null_if_missing(value: object) -> object:
    """Map pandas/NumPy missing values to None."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return value
    return value
