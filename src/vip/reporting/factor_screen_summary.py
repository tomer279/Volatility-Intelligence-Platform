"""Factor-screen report context assembly.

Exports
-------
ScreenReportPayload
    Minimal screen outputs needed by the report builder.
ReportIdentity
    Header identity fields for a factor-screen memo.
ReportTables
    Tabular sections for a factor-screen memo.
FactorScreenReportContext
    Render-ready context for the factor-screen report.
build_factor_screen_context
    Assemble a report context from payload, plot path, and meta.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from datetime import date

import pandas as pd

from vip.domain.errors import DataValidationError

from vip.reporting.report_common import (
    DEFAULT_CAVEATS,
    ReportMeta,
    format_oos_gap_wording,
    frame_records,
)
from vip.reporting.report_sections import (
    ReportExtras,
    build_implied_vs_realized_section,
    build_parametric_vs_har_section,
    implied_section_template_payload,
    parametric_section_template_payload,
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
        Importance image, caveats, and Implied / Parametric section inputs.

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
        payload.update(
            implied_section_template_payload(
                self.extras.implied_vs_realized
            )
        )
        payload.update(
            parametric_section_template_payload(
                self.extras.parametric_vs_har
            )
        )
        return payload


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
    factor_rows = frame_records(payload.ranking)
    implied = build_implied_vs_realized_section(model_rows, factor_rows)
    parametric = build_parametric_vs_har_section(model_rows)
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
            parametric_vs_har=parametric,
        ),
    )


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
    rows = frame_records(summary)
    for row in rows:
        row["comparison_note"] = format_oos_gap_wording(row)
    return rows
