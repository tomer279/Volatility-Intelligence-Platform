# `vip.reporting`

## Purpose
Render reproducible research memos from experiment artifacts, including
locked OOS-gap wording and inference methodology fields.

## Modules
- `experiment_summary.py` - Template context builders; `InferenceReportMeta`;
  `format_oos_gap_wording`; inference caveats (bootstrap primary; non-overlap footnote).
- `html_report.py` - Jinja2 HTML report renderer.
- `templates/factor_screen.html.j2` - Factor-screen memo with inference-enriched horse-race.

## Key APIs
- `InferenceReportMeta` - Baseline, NW lags, block length, α, bootstrap resamples.
- `ReportMeta` - Locked methodology fields (target, splits, embargo, inference).
- `ScreenReportPayload` - Summary / ranking / regime tables for the memo.
- `build_factor_screen_context(payload, plot_path, meta)` - Render-ready context.
- `format_oos_gap_wording(row)` - “Significantly lower…” only when primary bootstrap
  rejects at α and mean ΔQLIKE < 0; otherwise descriptive gap wording.
- `render_factor_screen_report(context)` / `write_html_report(path, html)` - Render + persist.

## Notes
- Reports should include methodology caveats, not only metrics tables.
- Horse-race QLIKE rankings without inference are descriptive, not findings.
- Embargo blocks leakage; it is not a significance test.
- Non-overlapping every-horizon bootstrap is a footnote only, not a second primary claim.
- The "What works when" section shows the best model per regime (min QLIKE).
- Keep CLI thin; call reporting from application use-cases.
