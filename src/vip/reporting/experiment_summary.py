"""Build template contexts for experiment reports.

Exports
-------
InferenceReportMeta
    Inference fields shown in the locked-methodology list.
ReportMeta
    Locked methodology fields shown in the HTML memo.
ScreenReportPayload
    Minimal screen outputs needed by the report builder.
ReportIdentity
    Header identity fields for a factor-screen memo.
ReportTables
    Tabular sections for a factor-screen memo.
ImpliedVsRealizedSection
    Context for the HTML “Implied vs realized” memo block.
ReportExtras
    Non-tabular extras for a factor-screen memo (image, caveats, Implied section).
FactorScreenReportContext
    Render-ready context for the factor-screen report.
format_oos_gap_wording
    Locked comparison text gated on primary bootstrap significance.
build_implied_vs_realized_section
    Derive IV-model row + optional gap-feature rows from screen tables.
build_factor_screen_contextןצפךןקגהדרקשך
    Assemble a report context from payload, plot path, and meta.
MultiHorizonReportMeta
    Study-level methodology fields for multi-horizon memos.
MultiHorizonReportPayload
    Cross-horizon summary table for the study memo.
MultiHorizonReportContext
    Render-ready context for the multi-horizon report.
build_multi_horizon_context
    Assemble a multi-horizon report context.
MultiHorizonReportIdentity
    Header identity fields for a multi-horizon memo.
MultiHorizonReportTables
    Skill-by-horizon and per-horizon methodology rows.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from vip.domain.errors import DataValidationError

DEFAULT_TARGET_COLUMN = "target_rv_cc_5d"
DEFAULT_PRIMARY_METRIC = "qlike"
INFERENCE_CAVEATS = (
    "Horse-race QLIKE rankings without inference are descriptive, not findings.",
    "Overlapping multi-day RV labels induce dependence in loss differentials; "
    "primary inference is a block bootstrap of mean OOS ΔQLIKE vs HAR.",
    "Walk-forward embargo blocks train/test leakage; it is not a significance test.",
    "Say 'significantly better' only when the primary bootstrap rejects at alpha "
    "and mean ΔQLIKE favors the challenger; otherwise 'lower mean OOS QLIKE'.",
    "Non-overlapping every-horizon-day bootstrap is a footnote sensitivity check, "
    "not a second primary significance test.",
)
DEFAULT_CAVEATS = (
    "HAR lags (rv_cc_1d / rv_cc_5d / rv_cc_21d) are collinear; treat them as a "
    "feature family, not independent discoveries.",
    "Permutation importance is associative, not causal.",
    "Rankings can be unstable for weak factors and may shift across regimes.",
    "Results are for a single liquid ETF sample (SPY MVP) and should not be "
    "over-generalized.",
    "QLIKE permutation deltas can spike on collinear HAR lags; rankings use "
    "median importance across folds to limit single-fold domination.",
) + INFERENCE_CAVEATS
MULTI_HORIZON_CAVEATS = DEFAULT_CAVEATS + (
    "Each horizon uses its own target, embargo, NW lags, and bootstrap block "
    "length; do not pool loss differentials across horizons.",
)

DEFAULT_BASELINE_MODEL = "har_rv_ols"
DEFAULT_NW_LAGS = 4
DEFAULT_BOOTSTRAP_BLOCK_LENGTH = 15
DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_N_RESAMPLES = 1999
VIX_AS_FORECAST_MODEL = "vix_as_forecast"
IV_RV_GAP_FEATURE_PREFIX = "vix_minus_rv_"
DEFAULT_TOP_IV_RV_GAP_FEATURES = 3
VIX_PROXY_CAVEAT = (
    "VIX is used as an IV proxy for index/ETF research; it is not "
    "single-name implied volatility and is not a variance-swap or "
    "options-replication identity."
)
VIX_UNIT_CONVERSION_NOTE = (
    "Unit conversion (locked): "
    "vix_vol_daily = (vix_level / 100.0) / sqrt(252), "
    "putting VIX on the same non-annualized daily-vol scale as rv_cc_*."
)

@dataclass(frozen=True, slots=True)
class InferenceReportMeta:
    """Inference fields shown in the locked-methodology list.

    Parameters
    ----------
    baseline_model : str
        Horse-race reference model.
    nw_lags : int
        Newey–West lag (horizon − 1).
    bootstrap_block_length : int
        Primary block-bootstrap length.
    alpha : float
        Two-sided significance level.
    bootstrap_n_resamples : int
        Bootstrap replication count.

    Methods
    -------
    validate()
        Raise if fields are invalid.
    describe()
        Return a short human-readable summary.
    """
    baseline_model: str = DEFAULT_BASELINE_MODEL
    nw_lags: int = DEFAULT_NW_LAGS
    bootstrap_block_length: int = DEFAULT_BOOTSTRAP_BLOCK_LENGTH
    alpha: float = DEFAULT_ALPHA
    bootstrap_n_resamples: int = DEFAULT_BOOTSTRAP_N_RESAMPLES

    def validate(self) -> None:
        """Raise ``DataValidationError`` when inference meta is invalid."""
        if not self.baseline_model.strip():
            raise DataValidationError("baseline_model must be non-empty.")
        if self.nw_lags < 0:
            raise DataValidationError("nw_lags must be non-negative.")
        if self.bootstrap_block_length < 1:
            raise DataValidationError("bootstrap_block_length must be >= 1.")
        if not 0.0 < self.alpha < 1.0:
            raise DataValidationError("alpha must be in (0, 1).")
        if self.bootstrap_n_resamples < 1:
            raise DataValidationError("bootstrap_n_resamples must be >= 1.")

    def describe(self) -> str:
        """Return a short human-readable summary."""
        return (
            f"baseline={self.baseline_model}, nw_lags={self.nw_lags}, "
            f"block={self.bootstrap_block_length}, alpha={self.alpha}"
        )


@dataclass(frozen=True, slots=True)
class ReportMeta:
    """Locked methodology fields for the factor-screen memo.

    Parameters
    ----------
    target_column : str
        Target column name.
    primary_metric : str
        Primary evaluation metric name.
    n_splits : int
        Number of walk-forward folds.
    embargo_size : int
        Embargo length in sessions.
    inference : InferenceReportMeta
        Bootstrap / NW / baseline fields for the methodology list.

    Methods
    -------
    validate()
        Raise if metadata is invalid.
    describe()
        Return a short human-readable summary.
    """

    target_column: str = DEFAULT_TARGET_COLUMN
    primary_metric: str = DEFAULT_PRIMARY_METRIC
    n_splits: int = 5
    embargo_size: int = 5
    inference: InferenceReportMeta = field(default_factory=InferenceReportMeta)

    def validate(self) -> None:
        """Raise ``DataValidationError`` when metadata is invalid."""
        if not self.target_column.strip():
            raise DataValidationError("target_column must be non-empty.")
        if not self.primary_metric.strip():
            raise DataValidationError("primary_metric must be non-empty.")
        if self.n_splits < 2:
            raise DataValidationError("n_splits must be at least 2.")
        if self.embargo_size < 0:
            raise DataValidationError("embargo_size must be non-negative.")
        self.inference.validate()

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact methodology summary.
        """
        return (
            f"{self.target_column} / {self.primary_metric} / "
            f"n_splits={self.n_splits} / embargo={self.embargo_size} / "
            f"{self.inference.describe()}"
        )


@dataclass(frozen=True, slots=True)
class ScreenReportPayload:
    """Minimal screen outputs needed by the report builder.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    experiment_id : str
        Experiment identifier.
    screening_model : str
        Model used for permutation importance.
    summary : pandas.DataFrame
        Model horse-race table.
    ranking : pandas.DataFrame
        Factor ranking table.
    regime_metrics : pandas.DataFrame
        Regime-sliced out-of-sample metrics with columns like
        ``regime``, ``model``, ``n_obs``, ``qlike``, ``mse``, and ``mae``.

    Methods
    -------
    model_count()
        Return the number of model rows.
    factor_count()
        Return the number of factor rows.
    """

    symbol: str
    experiment_id: str
    screening_model: str
    summary: pd.DataFrame
    ranking: pd.DataFrame
    regime_metrics : pd.DataFrame

    def model_count(self) -> int:
        """Return the number of model rows.

        Returns
        -------
        int
            Row count of ``summary``.
        """
        return int(len(self.summary))

    def factor_count(self) -> int:
        """Return the number of factor rows.

        Returns
        -------
        int
            Row count of ``ranking``.
        """
        return int(len(self.ranking))


@dataclass(frozen=True, slots=True)
class ReportIdentity:
    """Header identity fields for a factor-screen memo.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    experiment_id : str
        Experiment identifier.
    generated_on : str
        Generation date (ISO format).
    screening_model : str
        Model used for permutation importance.

    Methods
    -------
    describe()
        Return a short human-readable summary.
    as_dict()
        Return a flat mapping of identity fields.
    """

    symbol: str
    experiment_id: str
    generated_on: str
    screening_model: str

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact identity summary.
        """
        return f"{self.symbol} / {self.experiment_id}"

    def as_dict(self) -> dict[str, str]:
        """Return a flat mapping of identity fields.

        Returns
        -------
        dict of str to str
            Identity fields for template rendering.
        """
        return {
            "symbol": self.symbol,
            "experiment_id": self.experiment_id,
            "generated_on": self.generated_on,
            "screening_model": self.screening_model,
        }


@dataclass(frozen=True, slots=True)
class ReportTables:
    """Tabular sections for a factor-screen memo.

    Parameters
    ----------
    model_rows : list of dict
        Horse-race table rows.
    factor_rows : list of dict
        Factor-ranking table rows.
    regime_best_rows : list of dict
        One “best model” row per regime, suitable for the "What works when"
        table in the HTML report.

    Methods
    -------
    model_count()
        Return the number of model rows.
    factor_count()
        Return the number of factor rows.
    """

    model_rows: list[dict[str, object]]
    factor_rows: list[dict[str, object]]
    regime_best_rows: list[dict[str, object]]

    def model_count(self) -> int:
        """Return the number of model rows.

        Returns
        -------
        int
            Length of ``model_rows``.
        """
        return int(len(self.model_rows))

    def factor_count(self) -> int:
        """Return the number of factor rows.

        Returns
        -------
        int
            Length of ``factor_rows``.
        """
        return int(len(self.factor_rows))


@dataclass(frozen=True, slots=True)
class ImpliedVsRealizedSection:
    """Inputs for the HTML “Implied vs realized” section.

    Parameters
    ----------
    proxy_caveat : str
        VIX-as-IV-proxy research caveat.
    unit_conversion_note : str
        Locked ``vix_vol_daily`` formula note.
    vix_forecast_row : dict of str to object or None
        Horse-race row for ``vix_as_forecast`` (with ``comparison_note``),
        or ``None`` when the model was not screened.
    gap_feature_rows : tuple of dict
        Top IV−RV gap features from the ranking (may be empty).

    Methods
    -------
    has_vix_forecast()
        Return whether a VIX-as-forecast horse-race row is present.
    gap_feature_count()
        Return the number of gap-feature rows.
    """

    proxy_caveat: str
    unit_conversion_note: str
    vix_forecast_row: dict[str, object] | None
    gap_feature_rows: tuple[dict[str, object], ...]

    def has_vix_forecast(self) -> bool:
        """Return whether a VIX-as-forecast horse-race row is present."""
        return self.vix_forecast_row is not None

    def gap_feature_count(self) -> int:
        """Return the number of gap-feature rows."""
        return int(len(self.gap_feature_rows))


@dataclass(frozen=True, slots=True)
class ReportExtras:
    """Non-tabular extras for a factor-screen memo.

    Parameters
    ----------
    importance_image_base64 : str or None
        Base64-encoded PNG bytes, if available.
    caveats : tuple of str
        Research caveats shown in the memo.
    implied_vs_realized : ImpliedVsRealizedSection or None
        Optional Implied vs realized section inputs.

    Methods
    -------
    has_image()
        Return whether an importance image is present.
    caveat_count()
        Return the number of caveat strings.
    """

    importance_image_base64: str | None
    caveats: tuple[str, ...]
    implied_vs_realized: ImpliedVsRealizedSection | None = None

    def has_image(self) -> bool:
        """Return whether an importance image is present.

        Returns
        -------
        bool
            True when base64 image data exists.
        """
        return self.importance_image_base64 is not None

    def caveat_count(self) -> int:
        """Return the number of caveat strings.

        Returns
        -------
        int
            Length of ``caveats``.
        """
        return int(len(self.caveats))


@dataclass(frozen=True, slots=True)
class FactorScreenReportContext:
    """Render-ready context for the factor-screen HTML report.

    Parameters
    ----------
    identity : ReportIdentity
        Symbol, experiment id, date, and screening model.
    methodology : ReportMeta
        Locked target/metric/walk-forward fields.
    tables : ReportTables
        Model horse-race, factor ranking, and regime-best rows.
    extras : ReportExtras
        Importance image, caveats, and Implied vs realized section inputs.

    Methods
    -------
    as_template_dict()
        Return a flat mapping suitable for Jinja2 rendering.
    describe()
        Return a short human-readable summary.
    """

    identity: ReportIdentity
    methodology: ReportMeta
    tables: ReportTables
    extras: ReportExtras

    def as_template_dict(self) -> dict[str, object]:
        """Return a flat mapping suitable for Jinja2 rendering.

        Returns
        -------
        dict of str to object
            Template variables (flat keys preserved for the existing template).
        """
        payload: dict[str, object] = {}
        payload.update(self.identity.as_dict())
        payload.update(
            {
                "target_column": self.methodology.target_column,
                "primary_metric": self.methodology.primary_metric,
                "n_splits": self.methodology.n_splits,
                "embargo_size": self.methodology.embargo_size,
                "model_rows": self.tables.model_rows,
                "factor_rows": self.tables.factor_rows,
                "importance_image_base64": self.extras.importance_image_base64,
                "caveats": list(self.extras.caveats),
                "regime_best_rows": self.tables.regime_best_rows,
                "baseline_model": self.methodology.inference.baseline_model,
                "nw_lags": self.methodology.inference.nw_lags,
                "bootstrap_block_length": self.methodology.inference.bootstrap_block_length,
                "alpha": self.methodology.inference.alpha,
                "bootstrap_n_resamples": self.methodology.inference.bootstrap_n_resamples,
            }
        )
        implied = self.extras.implied_vs_realized
        if implied is not None:
            payload.update(
                {
                    "implied_proxy_caveat": implied.proxy_caveat,
                    "implied_unit_conversion_note": implied.unit_conversion_note,
                    "vix_forecast_row": implied.vix_forecast_row,
                    "iv_rv_gap_rows": list(implied.gap_feature_rows),
                }
            )
        else:
            payload.update(
                {
                    "implied_proxy_caveat": VIX_PROXY_CAVEAT,
                    "implied_unit_conversion_note": VIX_UNIT_CONVERSION_NOTE,
                    "vix_forecast_row": None,
                    "iv_rv_gap_rows": [],
                }
            )
        return payload

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact report-context summary.
        """
        return (
            f"{self.identity.describe()} / "
            f"models={self.tables.model_count()} / "
            f"factors={self.tables.factor_count()}"
        )


@dataclass(frozen=True, slots=True)
class MultiHorizonReportMeta:
    """Study-level methodology for a multi-horizon memo.

    Parameters
    ----------
    horizons : tuple of int
        Horizons included in the study.
    baseline_model : str
        Inference baseline name.
    alpha : float
        Two-sided significance level.
    per_horizon : tuple of dict
        Per-horizon embargo / NW / block-length rows.

    Methods
    -------
    validate()
        Raise if metadata is invalid.
    describe()
        Return a short human-readable summary.
    """

    horizons: tuple[int, ...]
    baseline_model: str = DEFAULT_BASELINE_MODEL
    alpha: float = DEFAULT_ALPHA
    per_horizon: tuple[dict[str, object], ...] = ()

    def validate(self) -> None:
        """Raise ``DataValidationError`` when metadata is invalid."""
        if not self.horizons:
            raise DataValidationError("horizons must be non-empty.")
        if not self.baseline_model.strip():
            raise DataValidationError("baseline_model must be non-empty.")
        if not 0.0 < self.alpha < 1.0:
            raise DataValidationError("alpha must be in (0, 1).")

    def describe(self) -> str:
        """Return a short human-readable summary."""
        joined = ",".join(str(h) for h in self.horizons)
        return f"horizons=[{joined}], baseline={self.baseline_model}, alpha={self.alpha}"


@dataclass(frozen=True, slots=True)
class MultiHorizonReportPayload:
    """Minimal multi-horizon outputs for the study memo.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    study_id : str
        Study identifier.
    horizon_summary : pandas.DataFrame
        Stacked model × horizon table.

    Methods
    -------
    row_count()
        Return the number of summary rows.
    describe()
        Return a short human-readable summary.
    """

    symbol: str
    study_id: str
    horizon_summary: pd.DataFrame

    def row_count(self) -> int:
        """Return the number of summary rows."""
        return int(len(self.horizon_summary))

    def describe(self) -> str:
        """Return a short human-readable summary."""
        return f"{self.symbol} / {self.study_id} / rows={self.row_count()}"


@dataclass(frozen=True, slots=True)
class MultiHorizonReportIdentity:
    """Header identity fields for a multi-horizon memo.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    study_id : str
        Study identifier.
    generated_on : str
        Generation date (ISO format).
    horizons_label : str
        Display string for horizons (for example ``1, 5, 21``).

    Methods
    -------
    describe()
        Return a short human-readable summary.
    as_dict()
        Return a flat mapping of identity fields.
    """

    symbol: str
    study_id: str
    generated_on: str
    horizons_label: str

    def describe(self) -> str:
        """Return a short human-readable summary."""
        return f"{self.symbol} / {self.study_id}"

    def as_dict(self) -> dict[str, str]:
        """Return a flat mapping of identity fields."""
        return {
            "symbol": self.symbol,
            "study_id": self.study_id,
            "generated_on": self.generated_on,
            "horizons_label": self.horizons_label,
        }


@dataclass(frozen=True, slots=True)
class MultiHorizonReportTables:
    """Tabular sections for a multi-horizon memo.

    Parameters
    ----------
    horizon_rows : list of dict
        Skill-by-horizon table rows (with ``comparison_note``).
    per_horizon_rows : list of dict
        Per-horizon embargo / NW / block-length methodology rows.

    Methods
    -------
    row_count()
        Return the number of skill-by-horizon rows.
    methodology_row_count()
        Return the number of per-horizon methodology rows.
    """

    horizon_rows: list[dict[str, object]]
    per_horizon_rows: list[dict[str, object]]

    def row_count(self) -> int:
        """Return the number of skill-by-horizon rows."""
        return int(len(self.horizon_rows))

    def methodology_row_count(self) -> int:
        """Return the number of per-horizon methodology rows."""
        return int(len(self.per_horizon_rows))


@dataclass(frozen=True, slots=True)
class MultiHorizonReportContext:
    """Render-ready context for the multi-horizon HTML report.

    Parameters
    ----------
    identity : MultiHorizonReportIdentity
        Symbol, study id, date, and horizons label.
    methodology : MultiHorizonReportMeta
        Baseline, alpha, and locked horizon set.
    tables : MultiHorizonReportTables
        Skill-by-horizon and per-horizon methodology rows.
    extras : ReportExtras
        Caveats (importance image unused; pass ``None``).

    Methods
    -------
    as_template_dict()
        Return a flat mapping for Jinja2.
    describe()
        Return a short human-readable summary.
    """

    identity: MultiHorizonReportIdentity
    methodology: MultiHorizonReportMeta
    tables: MultiHorizonReportTables
    extras: ReportExtras

    def as_template_dict(self) -> dict[str, object]:
        """Return a flat mapping for Jinja2."""
        payload: dict[str, object] = {}
        payload.update(self.identity.as_dict())
        payload.update(
            {
                "baseline_model": self.methodology.baseline_model,
                "alpha": self.methodology.alpha,
                "per_horizon_rows": self.tables.per_horizon_rows,
                "horizon_rows": self.tables.horizon_rows,
                "caveats": list(self.extras.caveats),
            }
        )
        return payload

    def describe(self) -> str:
        """Return a short human-readable summary."""
        return (
            f"{self.identity.describe()} / "
            f"rows={self.tables.row_count()}"
        )


def build_multi_horizon_context(
        payload: MultiHorizonReportPayload,
        meta: MultiHorizonReportMeta,
) -> MultiHorizonReportContext:
    """Assemble a multi-horizon report context.

    Parameters
    ----------
    payload : MultiHorizonReportPayload
        Stacked horizon summary.
    meta : MultiHorizonReportMeta
        Study-level methodology fields.

    Returns
    -------
    MultiHorizonReportContext
        Render-ready template context.
    """
    meta.validate()
    horizon_rows = _frame_records(payload.horizon_summary)
    for row in horizon_rows:
        row["comparison_note"] = format_oos_gap_wording(row)
    joined = ", ".join(str(h) for h in meta.horizons)
    return MultiHorizonReportContext(
        identity=MultiHorizonReportIdentity(
            symbol=payload.symbol,
            study_id=payload.study_id,
            generated_on=date.today().isoformat(),
            horizons_label=joined,
        ),
        methodology=meta,
        tables=MultiHorizonReportTables(
            horizon_rows=horizon_rows,
            per_horizon_rows=list(meta.per_horizon),
        ),
        extras=ReportExtras(
            importance_image_base64=None,
            caveats=MULTI_HORIZON_CAVEATS,
        ),
    )


def format_oos_gap_wording(row: dict[str, object]) -> str:
    """Return locked comparison text for one horse-race row.

    Parameters
    ----------
    row : dict of str to object
        Summary row; may include inference columns.

    Returns
    -------
    str
        Baseline label, or gap wording gated on bootstrap significance.
    """
    mean_delta = row.get("mean_delta_qlike")
    if mean_delta is None or (isinstance(mean_delta, float) and pd.isna(mean_delta)):
        return "baseline (reference)"

    significant = bool(row.get("significant_vs_baseline"))
    delta = float(mean_delta)
    if significant and delta < 0.0:
        return "significantly lower mean OOS QLIKE vs HAR (bootstrap)"
    if delta < 0.0:
        return "lower mean OOS QLIKE vs HAR (not significant at α)"
    if delta > 0.0:
        return "higher mean OOS QLIKE vs HAR (not significant at α)"
    return "similar mean OOS QLIKE vs HAR"


def build_factor_screen_context(
        payload: ScreenReportPayload,
        plot_path: Path | None,
        meta: ReportMeta,
) -> FactorScreenReportContext:
    """Assemble a report context from payload, plot path, and meta.

    Parameters
    ----------
    payload : ScreenReportPayload
        Must include ``regime_metrics`` (used for the "What works when" section).
    plot_path : pathlib.Path or None
        Optional path to ``importance_plot.png``.
    meta : ReportMeta
        Locked methodology fields.

    Returns
    -------
    FactorScreenReportContext
        Render-ready template context.

    Raises
    ------
    DataValidationError
        If ``meta`` is invalid.
    """
    meta.validate()
    regime_best_rows = _best_model_by_regime_rows(payload.regime_metrics)
    model_rows = _model_rows_with_wording(payload.summary)
    factor_rows = _frame_records(payload.ranking)
    implied = build_implied_vs_realized_section(model_rows, factor_rows)
    return FactorScreenReportContext(
        identity=ReportIdentity(
            symbol=payload.symbol,
            experiment_id=payload.experiment_id,
            generated_on=date.today().isoformat(),
            screening_model=payload.screening_model,
        ),
        methodology=meta,
        tables=ReportTables(
            model_rows=model_rows,
            factor_rows=factor_rows,
            regime_best_rows=regime_best_rows,
        ),
        extras=ReportExtras(
            importance_image_base64=_encode_png(plot_path),
            caveats=DEFAULT_CAVEATS,
            implied_vs_realized=implied,
        ),
    )


def build_implied_vs_realized_section(
        model_rows: list[dict[str, object]],
        factor_rows: list[dict[str, object]],
) -> ImpliedVsRealizedSection:
    """Derive Implied vs realized inputs from horse-race and ranking rows.

    Parameters
    ----------
    model_rows : list of dict
        Horse-race rows after ``format_oos_gap_wording``.
    factor_rows : list of dict
        Factor-ranking rows (already importance-ordered).

    Returns
    -------
    ImpliedVsRealizedSection
        Section context; ``vix_forecast_row`` may be ``None``.
    """
    forecast_row = _find_model_row(model_rows, VIX_AS_FORECAST_MODEL)
    gap_rows = _top_iv_rv_gap_feature_rows(
        factor_rows,
        DEFAULT_TOP_IV_RV_GAP_FEATURES,
    )
    return ImpliedVsRealizedSection(
        proxy_caveat=VIX_PROXY_CAVEAT,
        unit_conversion_note=VIX_UNIT_CONVERSION_NOTE,
        vix_forecast_row=forecast_row,
        gap_feature_rows=tuple(gap_rows),
    )


def _find_model_row(
        model_rows: list[dict[str, object]],
        model_name: str,
) -> dict[str, object] | None:
    """Return the first horse-race row matching ``model_name``, else None."""
    for row in model_rows:
        if str(row.get("model", "")) == model_name:
            return row
    return None


def _top_iv_rv_gap_feature_rows(
        factor_rows: list[dict[str, object]],
        limit: int,
) -> list[dict[str, object]]:
    """Keep ranking rows whose feature name starts with ``vix_minus_rv_``."""
    matched = [
        row
        for row in factor_rows
        if str(row.get("feature", "")).startswith(IV_RV_GAP_FEATURE_PREFIX)
    ]
    return matched[:limit]


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to JSON-like row mappings (NaN → None)."""
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        records.append({key: _null_if_missing(value) for key, value in row.items()})
    return records


def _null_if_missing(value: object) -> object:
    """Map pandas/NumPy missing values to None for Jinja."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return value
    return value


def _best_model_by_regime_rows(
        regime_metrics: pd.DataFrame
    ) -> list[dict[str, object]]:
    """Pick the best (min qlike) model per regime; keep empty-regime placeholder."""
    required = {"regime", "model", "n_obs", "qlike", "mse", "mae"}
    missing = required.difference(regime_metrics.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise DataValidationError(
            f"regime_metrics missing required columns: {missing_text}."
        )

    # keep regime order stable (full_sample then covid then bear)
    regime_order = list(regime_metrics["regime"].drop_duplicates())

    rows: list[dict[str, object]] = []
    for regime in regime_order:
        group = regime_metrics[regime_metrics["regime"] == regime]

        valid = group.dropna(subset=["qlike"])
        if not valid.empty:
            best = valid.sort_values("qlike", ascending=True).iloc[0]
        else:
            # empty slice placeholder: still show something
            best = group.iloc[0]

        rows.append({
            "regime": str(regime),
            "model": str(best["model"]),
            "qlike": None if pd.isna(best["qlike"]) else float(best["qlike"]),
            "mse": None if pd.isna(best["mse"]) else float(best["mse"]),
            "mae": None if pd.isna(best["mae"]) else float(best["mae"]),
            "n_obs": int(best["n_obs"]),
        })
    return rows

def _encode_png(plot_path: Path | None) -> str | None:
    """Base64-encode a PNG file when present."""
    if plot_path is None or not plot_path.is_file():
        return None
    return base64.b64encode(plot_path.read_bytes()).decode("ascii")


def _model_rows_with_wording(summary: pd.DataFrame) -> list[dict[str, object]]:
    """Convert summary rows and attach locked comparison wording."""
    rows = _frame_records(summary)
    for row in rows:
        row["comparison_note"] = format_oos_gap_wording(row)
    return rows
