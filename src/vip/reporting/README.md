# `vip.reporting`

## Purpose

Render reproducible research memos from experiment artifacts.

## Modules

- `html_report.py` - Jinja2 HTML report renderer (Step 8).

- `experiment_summary.py` - Template context builders (Step 8).

- `templates/` - HTML templates.

## Notes

- Reports should include methodology caveats, not only metrics tables.
- The "What works when" section shows the best model per regime (min QLIKE).
- Keep CLI thin; call reporting from application use-cases.