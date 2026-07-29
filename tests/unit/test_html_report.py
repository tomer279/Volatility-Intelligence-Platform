"""Tests for factor-screen HTML report rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vip.reporting.experiment_summary import (
    ReportMeta,
    ScreenReportPayload,
    build_factor_screen_context,
)
from vip.reporting.html_report import render_factor_screen_report, write_html_report


def _payload() -> ScreenReportPayload:
    """Build a tiny screen payload for report tests."""
    summary = pd.DataFrame(
        {
            "model": ["ridge", "har_rv_ols"],
            "qlike": [0.11, 0.12],
            "mse": [0.01, 0.02],
            "mae": [0.05, 0.06],
        }
    )
    ranking = pd.DataFrame(
        {
            "feature": ["rv_cc_5d", "rv_cc_1d"],
            "mean_importance": [0.4, 0.2],
            "std_importance": [0.05, 0.03],
            "top_k_hit_rate": [1.0, 0.5],
            "n_folds": [3, 3],
        }
    )
    return ScreenReportPayload(
        symbol="SPY",
        experiment_id="factor-screen-spy-test",
        screening_model="ridge",
        summary=summary,
        ranking=ranking,
    )


def test_render_factor_screen_report_contains_sections() -> None:
    """HTML memo should include core research sections."""
    context = build_factor_screen_context(
        payload=_payload(),
        plot_path=None,
        meta=ReportMeta(n_splits=3, embargo_size=5),
    )
    html = render_factor_screen_report(context)
    for snippet in (
        "Factor Screen",
        "Research question",
        "Locked methodology",
        "Model horse-race",
        "Ranked factors",
        "Caveats",
        "target_rv_cc_5d",
        "ridge",
        "rv_cc_5d",
    ):
        assert snippet in html


def test_write_html_report(tmp_path: Path) -> None:
    """Renderer should persist a non-empty HTML file."""
    context = build_factor_screen_context(
        payload=_payload(),
        plot_path=None,
        meta=ReportMeta(),
    )
    html = render_factor_screen_report(context)
    path = write_html_report(tmp_path / "report.html", html)
    assert path.is_file()
    assert "Caveats" in path.read_text(encoding="utf-8")