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
"""

from vip.reporting.experiment_summary import (
    FactorScreenReportContext,
    ReportMeta,
    ScreenReportPayload,
    build_factor_screen_context,
)
from vip.reporting.html_report import render_factor_screen_report, write_html_report

__all__ = [
    "FactorScreenReportContext",
    "ReportMeta",
    "ScreenReportPayload",
    "build_factor_screen_context",
    "render_factor_screen_report",
    "write_html_report",
]