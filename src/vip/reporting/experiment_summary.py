"""Build template contexts for experiment reports.

Exports
-------
ReportMeta
    Locked methodology fields shown in the HTML memo.
ScreenReportPayload
    Minimal screen outputs needed by the report builder.
ReportIdentity
    Header identity fields for a factor-screen memo.
ReportTables
    Tabular sections for a factor-screen memo.
ReportExtras
    Non-tabular extras for a factor-screen memo.
FactorScreenReportContext
    Render-ready context for the factor-screen report.
build_factor_screen_context
    Assemble a report context from payload, plot path, and meta.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from vip.domain.errors import DataValidationError

DEFAULT_TARGET_COLUMN = "target_rv_cc_5d"
DEFAULT_PRIMARY_METRIC = "qlike"
DEFAULT_CAVEATS = (
    "HAR lags (rv_cc_1d / rv_cc_5d / rv_cc_21d) are collinear; treat them as a "
    "feature family, not independent discoveries.",
    "Permutation importance is associative, not causal.",
    "Rankings can be unstable for weak factors and may shift across regimes.",
    "Results are for a single liquid ETF sample (SPY MVP) and should not be "
    "over-generalized.",
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

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact methodology summary.
        """
        return (
            f"{self.target_column} / {self.primary_metric} / "
            f"n_splits={self.n_splits} / embargo={self.embargo_size}"
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

    Methods
    -------
    model_count()
        Return the number of model rows.
    factor_count()
        Return the number of factor rows.
    """

    model_rows: list[dict[str, object]]
    factor_rows: list[dict[str, object]]

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
class ReportExtras:
    """Non-tabular extras for a factor-screen memo.

    Parameters
    ----------
    importance_image_base64 : str or None
        Base64-encoded PNG bytes, if available.
    caveats : tuple of str
        Research caveats shown in the memo.

    Methods
    -------
    has_image()
        Return whether an importance image is present.
    caveat_count()
        Return the number of caveat strings.
    """

    importance_image_base64: str | None
    caveats: tuple[str, ...]

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
        Model horse-race and factor-ranking rows.
    extras : ReportExtras
        Importance image and caveats.

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


def build_factor_screen_context(
    payload: ScreenReportPayload,
    plot_path: Path | None,
    meta: ReportMeta,
) -> FactorScreenReportContext:
    """Assemble a report context from payload, plot path, and meta.

    Parameters
    ----------
    payload : ScreenReportPayload
        Minimal screen outputs.
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
    return FactorScreenReportContext(
        identity=ReportIdentity(
            symbol=payload.symbol,
            experiment_id=payload.experiment_id,
            generated_on=date.today().isoformat(),
            screening_model=payload.screening_model,
        ),
        methodology=meta,
        tables=ReportTables(
            model_rows=_frame_records(payload.summary),
            factor_rows=_frame_records(payload.ranking),
        ),
        extras=ReportExtras(
            importance_image_base64=_encode_png(plot_path),
            caveats=DEFAULT_CAVEATS,
        ),
    )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to JSON-like row mappings."""
    return frame.to_dict(orient="records")


def _encode_png(plot_path: Path | None) -> str | None:
    """Base64-encode a PNG file when present."""
    if plot_path is None or not plot_path.is_file():
        return None
    return base64.b64encode(plot_path.read_bytes()).decode("ascii")
