"""Tests for factor-screen HTML report rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vip.reporting.experiment_summary import (
    ReportMeta,
    ScreenReportPayload,
    build_factor_screen_context,
    InferenceReportMeta,
    MultiHorizonReportMeta,
    MultiHorizonReportPayload,
    build_multi_horizon_context,
)
from vip.reporting.html_report import (
    render_factor_screen_report,
    write_html_report,    
    render_multi_horizon_screen_report
)

def _payload() -> ScreenReportPayload:
    """Build a tiny screen payload for report tests."""
    summary = pd.DataFrame(
        {
            "model": [
                "ridge",
                "vix_as_forecast",
                "ou_rv",
                "ewma_recursive",
                "har_rv_ols",
            ],
            "qlike": [0.11, 0.115, 0.118, 0.119, 0.12],
            "mse": [0.01, 0.012, 0.013, 0.014, 0.02],
            "mae": [0.05, 0.055, 0.057, 0.058, 0.06],
            "mean_delta_qlike": [-0.01, -0.005, 0.002, 0.003, None],
            "bootstrap_ci_low": [-0.02, -0.015, -0.01, -0.008, None],
            "bootstrap_ci_high": [-0.005, 0.001, 0.01, 0.012, None],
            "bootstrap_pvalue": [0.03, 0.12, 0.40, 0.45, None],
            "significant_vs_baseline": [True, False, False, False, None],
            "hln_pvalue": [0.04, 0.20, 0.50, 0.55, None],
        }
    )
    ranking = pd.DataFrame(
        {
            "feature": ["rv_cc_5d", "vix_minus_rv_5d", "rv_cc_1d"],
            "median_importance": [0.4, 0.25, 0.2],
            "mean_importance": [0.4, 0.24, 0.2],
            "std_importance": [0.05, 0.04, 0.03],
            "top_k_hit_rate": [1.0, 0.67, 0.5],
            "n_folds": [3, 3, 3],
        }
    )
    regime_metrics = pd.DataFrame(
        {
            "regime": ["full_sample", "covid_crash", "bear_2022"],
            "model": ["ridge", "har_rv_ols", "ridge"],
            "n_obs": [10, 3, 4],
            "qlike": [0.10, 0.08, 0.12],
            "mse": [0.01, 0.009, 0.011],
            "mae": [0.05, 0.04, 0.06],
        }
    )
    return ScreenReportPayload(
        symbol="SPY",
        experiment_id="factor-screen-spy-test",
        screening_model="ridge",
        summary=summary,
        ranking=ranking,
        regime_metrics=regime_metrics,
    )


def test_render_factor_screen_report_contains_sections() -> None:
    """HTML memo should include core research sections."""
    meta = ReportMeta(
        n_splits=3,
        embargo_size=5,
        inference=InferenceReportMeta(
            baseline_model="har_rv_ols",
            nw_lags=4,
            bootstrap_block_length=15,
            alpha=0.05,
        ),
    )
    html = render_factor_screen_report(build_factor_screen_context(_payload(), None, meta))
    assert "nan" not in html.lower()
    assert "—" in html
    for snippet in (
        "mean ΔQLIKE",
        "bootstrap CI",
        "bootstrap p",
        "significantly lower mean OOS QLIKE vs HAR (bootstrap)",
        "block length=15",
        "Newey–West lags=4",
        "har_rv_ols",
        "Implied vs realized",
        "vix_as_forecast",
        "vix_vol_daily = (vix_level / 100.0) / sqrt(252)",
        "vix_minus_rv_5d",
        "lower mean OOS QLIKE vs HAR (not significant at α)",
        "Parametric vs HAR",
        "ou_rv",
        "Discrete OU / AR(1) on log realized-vol target",
        "freezes the end-of-train log state",
        "not continuous-time SV",
        "higher mean OOS QLIKE vs HAR (not significant at α)",
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


def test_render_multi_horizon_skill_by_horizon_section() -> None:
    """Study memo must include Skill by horizon columns."""
    summary = pd.DataFrame(
        {
            "horizon_days": [1, 1, 5],
            "model": ["ridge", "har_rv_ols", "ridge"],
            "qlike": [0.10, 0.12, 0.11],
            "mse": [0.01, 0.02, 0.015],
            "mae": [0.05, 0.06, 0.055],
            "mean_delta_qlike": [-0.02, None, -0.01],
            "bootstrap_ci_low": [-0.03, None, -0.02],
            "bootstrap_ci_high": [-0.01, None, 0.0],
            "bootstrap_pvalue": [0.02, None, 0.08],
            "significant_vs_baseline": [True, None, False],
        }
    )
    payload = MultiHorizonReportPayload(
        symbol="SPY",
        study_id="multi-horizon-screen-spy-test",
        horizon_summary=summary,
    )
    meta = MultiHorizonReportMeta(
        horizons=(1, 5, 21),
        per_horizon=(
            {
                "horizon_days": 1,
                "target_column": "target_rv_cc_1d",
                "embargo_size": 1,
                "nw_lags": 0,
                "bootstrap_block_length": 10,
            },
            {
                "horizon_days": 5,
                "target_column": "target_rv_cc_5d",
                "embargo_size": 5,
                "nw_lags": 4,
                "bootstrap_block_length": 15,
            },
        ),
    )
    html = render_multi_horizon_screen_report(
        build_multi_horizon_context(payload, meta)
    )
    assert "Skill by horizon" in html
    assert "significant_vs_baseline" in html
    assert "horizon_days" in html
    assert "significantly lower mean OOS QLIKE vs HAR (bootstrap)" in html
    assert "nan" not in html.lower()


def test_implied_section_absent_forecast_still_renders_heading() -> None:
    """Section heading remains even when the IV model row is missing."""
    payload = _payload()
    summary = payload.summary.loc[payload.summary["model"] != "vix_as_forecast"]
    slim = ScreenReportPayload(
        symbol=payload.symbol,
        experiment_id=payload.experiment_id,
        screening_model=payload.screening_model,
        summary=summary,
        ranking=payload.ranking,
        regime_metrics=payload.regime_metrics,
    )
    html = render_factor_screen_report(
        build_factor_screen_context(slim, None, ReportMeta())
    )
    assert "Implied vs realized" in html
    assert "lacked" in html.lower() or "not in this horse-race" in html


def test_parametric_section_absent_ou_still_renders_heading() -> None:
    """Section heading and caveats remain when parametric models are missing."""
    payload = _payload()
    summary = payload.summary.loc[
        ~payload.summary["model"].isin(["ou_rv", "ewma_recursive"])
    ]
    slim = ScreenReportPayload(
        symbol=payload.symbol,
        experiment_id=payload.experiment_id,
        screening_model=payload.screening_model,
        summary=summary,
        ranking=payload.ranking,
        regime_metrics=payload.regime_metrics,
    )
    html = render_factor_screen_report(
        build_factor_screen_context(slim, None, ReportMeta())
    )
    assert "Parametric vs HAR" in html
    assert "Discrete OU / AR(1) on log realized-vol target" in html
    assert "freezes the end-of-train log state" in html
    assert "not in this horse-race" in html


def test_build_parametric_vs_har_section_finds_ou_rv() -> None:
    """Builder keeps ou_rv row and leaves stretch filter None."""
    from vip.reporting.experiment_summary import (
        build_parametric_vs_har_section,
        format_oos_gap_wording,
    )
    rows = [
        {
            "model": "ou_rv",
            "qlike": 0.118,
            "mean_delta_qlike": 0.002,
            "significant_vs_baseline": False,
        },
        {
            "model": "har_rv_ols",
            "qlike": 0.12,
            "mean_delta_qlike": None,
            "significant_vs_baseline": None,
        },
    ]
    for row in rows:
        row["comparison_note"] = format_oos_gap_wording(row)
    section = build_parametric_vs_har_section(rows)
    assert section.has_ou_rv()
    assert not section.has_filter_model()
    assert section.ou_rv_row is not None
    assert section.ou_rv_row["model"] == "ou_rv"
    assert "higher mean OOS QLIKE" in str(section.ou_rv_row["comparison_note"])
    assert any("not continuous-time SV" in c for c in section.caveats)


def test_build_parametric_vs_har_section_finds_ewma_recursive() -> None:
    """Builder attaches stretch filter row when ewma_recursive is present."""
    from vip.reporting.experiment_summary import (
        build_parametric_vs_har_section,
        format_oos_gap_wording,
    )
    rows = [
        {
            "model": "ou_rv",
            "qlike": 0.118,
            "mean_delta_qlike": 0.002,
            "significant_vs_baseline": False,
        },
        {
            "model": "ewma_recursive",
            "qlike": 0.119,
            "mean_delta_qlike": 0.003,
            "significant_vs_baseline": False,
        },
        {
            "model": "har_rv_ols",
            "qlike": 0.12,
            "mean_delta_qlike": None,
            "significant_vs_baseline": None,
        },
    ]
    for row in rows:
        row["comparison_note"] = format_oos_gap_wording(row)
    section = build_parametric_vs_har_section(rows)
    assert section.has_filter_model()
    assert section.filter_row is not None
    assert section.filter_row["model"] == "ewma_recursive"