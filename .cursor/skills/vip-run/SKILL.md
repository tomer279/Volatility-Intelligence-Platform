---
name: vip-run
description: >-
  Installs and runs the Volatility Intelligence Platform CLI (`vip`) with
  flagship defaults or user overrides. Use when the user wants to run vip,
  install the package, reproduce the SPY/QQQ study, screen factors,
  run multi-horizon screens, use VIX/jump features, or open experiment reports.
---

# VIP run

## Prerequisites
- Python 3.11+
- From repo root: `pip install -e ".[all]"` (or `.` plus needed extras)
- Network for first ingest (`yfinance`)

## Default happy path
Unless the user overrides, run:

```bash
vip run --symbols SPY,QQQ --with vix
```

- First run needs network; may take several minutes.
- Outputs: `data/artifacts/<experiment-id>/report.html` (CLI also prints the path).
- `vip run --with vix` ingests VIX automatically; step-by-step `vip features` needs `vip ingest --symbol VIX` first.

## Other common workflows

**Single symbol:**
```bash
vip run --symbol SPY --with vix
```

**Reuse existing data:**
```bash
vip run --symbol SPY --with vix --skip-ingest --skip-features
```

**Multi-horizon screen (1d / 5d / 21d):**
```bash
vip screen-horizons --symbol SPY --with vix
```
Artifacts: `data/artifacts/multi-horizon-screen-spy-<date>/report.html` and `horizon_summary.json`.

## Overrides
| User intent | CLI |
|-------------|-----|
| Symbols | `--symbol SPY` or `--symbols SPY,QQQ` |
| VIX | `--with vix` |
| Jump features | `--with jump` or `--with vix,jump` |
| No extras | omit `--with` |
| Skip ingest / features | `--skip-ingest`, `--skip-features` |

Allowed `--with` tokens: `vix`, `jump`, `iv_rv`, `rates` (comma-separated).

For walk-forward knobs (`n-splits`, `embargo`, etc.), use `vip screen` or `vip evaluate` — not `vip run`.

## Guardrails
- Research / educational only — not investment advice
- Do not commit `data/raw`, `data/processed`, `data/artifacts/`, secrets, or `.venv`
- Prefer the `vip` CLI; do not reimplement the pipeline in ad-hoc scripts
- If install fails, fix the environment first; do not invent alternate entrypoints