"""Experiment report rendering.

Exports
-------
ReportMeta
    Locked methodology fields shown in the HTML memo.
ScreenReportPayload
    Minimal screen outputs needed by the report builder.
FactorScreenReportContext
    Render-ready context for the factor-screen report.
build_factor_screen_context
    Assemble a report context from payload, plot path, and meta.
render_factor_screen_report
    Render the factor-screen memo to an HTML string.
write_html_report
    Persist an HTML string to disk.
MultiHorizonReportMeta
    Study-level methodology for multi-horizon memos.
MultiHorizonReportPayload
    Cross-horizon summary table for the study memo.
MultiHorizonReportContext
    Render-ready context for the multi-horizon report.
build_multi_horizon_context
    Assemble a multi-horizon report context.
render_multi_horizon_screen_report
    Render the multi-horizon study memo to an HTML string.
"""

from vip.reporting.experiment_summary import (
    FactorScreenReportContext,
    MultiHorizonReportContext,
    MultiHorizonReportMeta,
    MultiHorizonReportPayload,
    ReportMeta,
    ScreenReportPayload,
    build_factor_screen_context,
    build_multi_horizon_context,
)
from vip.reporting.html_report import (
    render_factor_screen_report,
    render_multi_horizon_screen_report,
    write_html_report,
)

__all__ = [
    "FactorScreenReportContext",
    "ReportMeta",
    "ScreenReportPayload",
    "build_factor_screen_context",
    "render_factor_screen_report",
    "write_html_report",
    "MultiHorizonReportContext",
    "MultiHorizonReportMeta",
    "MultiHorizonReportPayload",
    "build_multi_horizon_context",
    "render_multi_horizon_screen_report",
]