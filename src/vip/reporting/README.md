# `vip.reporting`

## Purpose
Render reproducible research memos from experiment artifacts, including
locked OOS-gap wording and inference methodology fields.

## Modules
- `report_common.py` - Methodology meta (`InferenceReportMeta`, `ReportMeta`),
  caveats, `format_oos_gap_wording`, DataFrame → Jinja helpers.
- `report_sections.py` - Implied vs realized and Parametric vs HAR section
  inputs/builders; `ReportExtras`; Jinja payload flatteners.
- `factor_screen_summary.py` - Single-horizon factor-screen context assembly
  (`build_factor_screen_context`).
- `multi_horizon_summary.py` - Multi-horizon study memo context assembly
  (`build_multi_horizon_context`).
- `experiment_summary.py` - Compatibility re-exports for the modules above.
- `html_report.py` - Jinja2 HTML report renderer.
- `templates/factor_screen.html.j2` - Factor-screen memo with inference-enriched
  horse-race, **Implied vs realized**, and **Parametric vs HAR**.
- `templates/multi_horizon_screen.html.j2` - Multi-horizon study memo with
  “Skill by horizon”.

## Key APIs
- `InferenceReportMeta` - Baseline, NW lags, block length, α, bootstrap resamples.
- `ReportMeta` - Locked methodology fields (target, splits, embargo, inference).
- `ScreenReportPayload` - Summary / ranking / regime tables for the memo
  (no dedicated IV/OU fields; thematic sections are derived from these tables).
- `ImpliedVsRealizedSection` / `build_implied_vs_realized_section(model_rows, factor_rows)` -
  VIX-proxy caveat, locked unit-conversion note, optional `vix_as_forecast` row
  (with `comparison_note`), optional top `vix_minus_rv_*` gap rows.
- `ParametricVsHarSection` / `build_parametric_vs_har_section(model_rows)` -
  discrete-OU caveats; optional `ou_rv` row (with `comparison_note`); optional
  stretch filter row when `ewma_recursive` appears in the horse-race.
- `build_factor_screen_context(payload, plot_path, meta)` - Render-ready context
  including Implied and Parametric section inputs on `ReportExtras`.
- `format_oos_gap_wording(row)` - “Significantly lower…” only when primary bootstrap
  rejects at α and mean ΔQLIKE < 0; otherwise descriptive gap wording.
- `render_factor_screen_report(context)` / `write_html_report(path, html)` - Render + persist.
- `MultiHorizonReportMeta` / `MultiHorizonReportPayload` - Study-level methodology + summary table.
- `MultiHorizonReportIdentity` / `MultiHorizonReportTables` / `MultiHorizonReportContext` -
  Nested render-ready multi-horizon context (≤7 instance attrs on the context object).
- `build_multi_horizon_context(payload, meta)` - Assemble the multi-horizon context.
- `render_multi_horizon_screen_report(context)` - Render the study memo HTML.

## Notes
- Prefer importing from `vip.reporting.experiment_summary` (facade) or
  `vip.reporting` package exports; internal modules may move as memos grow.
- Reports should include methodology caveats, not only metrics tables.
- Horse-race QLIKE rankings without inference are descriptive, not findings.
- Embargo blocks leakage; it is not a significance test.
- Non-overlapping every-horizon bootstrap is a footnote only, not a second primary claim.
- The "What works when" section shows the best model per regime (min QLIKE).
- The factor-screen memo includes **Implied vs realized**: always show the VIX-proxy
  and unit-conversion notes; show the `vix_as_forecast` ΔQLIKE / bootstrap row when
  that model is in the summary; show gap-feature rows when ranking contains
  `vix_minus_rv_*`. Significance language reuses `format_oos_gap_wording` (unchanged from M7).
- The factor-screen memo includes **Parametric vs HAR**: short discrete-OU /
  log-state / frozen-origin caveats; `ou_rv` ΔQLIKE / bootstrap row when present;
  stretch filter row when `ewma_recursive` is in the summary. Significance
  language reuses `format_oos_gap_wording` (unchanged from M7).
- Multi-horizon memos reuse `format_oos_gap_wording` for “Skill by horizon” cells;
  significance wording is unchanged from M7.
- Keep CLI thin; call reporting from application use-cases.