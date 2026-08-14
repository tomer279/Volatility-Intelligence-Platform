"""Build template contexts for experiment reports.

Compatibility re-exports for the split reporting modules.

Exports
-------
InferenceReportMeta
ReportMeta
ScreenReportPayload
ReportIdentity
ReportTables
ImpliedVsRealizedSection
ParametricVsHarSection
ReportExtras
FactorScreenReportContext
format_oos_gap_wording
build_implied_vs_realized_section
build_parametric_vs_har_section
build_factor_screen_context
MultiHorizonReportMeta
MultiHorizonReportPayload
MultiHorizonReportContext
MultiHorizonReportIdentity
MultiHorizonReportTables
build_multi_horizon_context
"""

from vip.reporting.factor_screen_summary import (
    FactorScreenReportContext,
    ReportIdentity,
    ReportTables,
    ScreenReportPayload,
    build_factor_screen_context,
)
from vip.reporting.multi_horizon_summary import (
    MultiHorizonReportContext,
    MultiHorizonReportIdentity,
    MultiHorizonReportMeta,
    MultiHorizonReportPayload,
    MultiHorizonReportTables,
    build_multi_horizon_context,
)
from vip.reporting.report_common import (
    InferenceReportMeta,
    ReportMeta,
    format_oos_gap_wording,
)
from vip.reporting.report_sections import (
    ImpliedVsRealizedSection,
    ParametricVsHarSection,
    ReportExtras,
    build_implied_vs_realized_section,
    build_parametric_vs_har_section,
)

__all__ = [
    "InferenceReportMeta",
    "ReportMeta",
    "ScreenReportPayload",
    "ReportIdentity",
    "ReportTables",
    "ImpliedVsRealizedSection",
    "ParametricVsHarSection",
    "ReportExtras",
    "FactorScreenReportContext",
    "format_oos_gap_wording",
    "build_implied_vs_realized_section",
    "build_parametric_vs_har_section",
    "build_factor_screen_context",
    "MultiHorizonReportMeta",
    "MultiHorizonReportPayload",
    "MultiHorizonReportContext",
    "MultiHorizonReportIdentity",
    "MultiHorizonReportTables",
    "build_multi_horizon_context",
]
