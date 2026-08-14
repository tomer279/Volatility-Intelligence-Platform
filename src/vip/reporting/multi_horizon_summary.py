"""Multi-horizon study memo context assembly.

Exports
-------
MultiHorizonReportMeta
    Study-level methodology fields for multi-horizon memos.
MultiHorizonReportPayload
    Cross-horizon summary table for the study memo.
MultiHorizonReportIdentity
    Header identity fields for a multi-horizon memo.
MultiHorizonReportTables
    Skill-by-horizon and per-horizon methodology rows.
MultiHorizonReportContext
    Render-ready context for the multi-horizon report.
build_multi_horizon_context
    Assemble a multi-horizon report context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from vip.domain.errors import DataValidationError
from vip.reporting.report_common import (
    DEFAULT_ALPHA,
    DEFAULT_BASELINE_MODEL,
    DEFAULT_CAVEATS,
    format_oos_gap_wording,
    frame_records
)
from vip.reporting.report_sections import ReportExtras


MULTI_HORIZON_CAVEATS = DEFAULT_CAVEATS + (
    "Each horizon uses its own target, embargo, NW lags, and bootstrap block "
    "length; do not pool loss differentials across horizons.",
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
    horizon_rows = frame_records(payload.horizon_summary)
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
