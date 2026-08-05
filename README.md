# Volatility Intelligence Platform

Research platform for forecasting realized volatility and screening which factors
best predict it. The toolchain covers ingestion, feature engineering, walk-forward
evaluation, factor screening, and HTML research memos — designed like an internal
quant research tool. 
> Research / educational use only. Not investment advice. Market data comes from
> Yahoo Finance via `yfinance`, which is unofficial and may rate-limit or break
> without notice.

## Features

- Daily OHLCV ingestion (Yahoo Finance → Parquet)
- Realized-volatility targets and feature families (HAR, returns, range, volume, optional VIX)
- Walk-forward baseline evaluation (QLIKE primary; MSE / MAE secondary)
- Factor screening with importance rankings and HTML reports
- Multi-horizon factor screens across 1d / 5d / 21d (`vip screen-horizons`)
- One-command study pipeline (`vip run`)
- Optional FastAPI server for browsing experiment artifacts

## Requirements

- Python 3.11+
- Network access for the first data download (`yfinance`)

## Install

```bash
git clone https://github.com/tomer279/Volatility-Intelligence-Platform.git
cd Volatility-Intelligence-Platform
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the package (editable) plus optional extras as needed:

```bash
pip install -e .
# Contributors / tests:
pip install -e ".[dev]"
# Tree models + SHAP:
pip install -e ".[nonlinear]"
# Artifact API:
pip install -e ".[api]"
```

On Windows, if `python` is a Store stub, prefer `py -3.11 -m venv .venv` and
`py -m pip install -e ".[dev]"`.

## Quick start

### 1. Sanity check

```bash
vip --help
vip info
```

### 2. Flagship study (one command)

Ingest OHLCV, build feature matrices (including VIX when requested), run factor
screens, and write HTML reports:

```bash
vip run --symbols SPY,QQQ --with vix
```

Single-symbol shorthand:

```bash
vip run --symbol SPY --with vix
```

The first run needs network access and may take several minutes. When finished,
the CLI prints a summary table and the path to each `report.html`.

### 3. Inspect outputs

Artifacts land under `data/artifacts/<experiment-id>/`, including:

| File | Contents |
|------|----------|
| `report.html` | Research memo (open in a browser) |
| `metrics.json` | Walk-forward metrics |
| `importance.json` / `factor_ranking.json` | Factor importance |
| `importance_plot.png` | Importance chart |

Raw bars and feature matrices are stored under `data/raw/` and `data/processed/`
(these directories are gitignored).

### 4. Step-by-step pipeline

Same workflow broken into explicit commands:

```bash
vip ingest --symbol SPY --start 2018-01-01 --end 2024-12-31
vip ingest --symbol VIX --start 2018-01-01 --end 2024-12-31
vip features --symbol SPY --with vix
vip evaluate --symbol SPY
vip screen --symbol SPY
```

`--with vix` on `features` requires VIX OHLCV already in `data/raw/`
(`vip ingest --symbol VIX`). `vip run --with vix` handles that automatically.
Allowed `--with` tokens: `vix`, `jump` (comma-separated, e.g. `vix,jump`).

### 5. Multi-horizon screen (1 / 5 / 21)

Requires OHLCV (and VIX if requested) already ingested. Rebuilds a feature
matrix per horizon unless `--skip-features`:

```bash
vip ingest --symbol SPY
vip ingest --symbol VIX
vip screen-horizons --symbol SPY --with vix
```

Open:

- `data/artifacts/multi-horizon-screen-spy-<date>/horizon_summary.json`
- `data/artifacts/multi-horizon-screen-spy-<date>/report.html`

Single-horizon screening remains `vip screen` (default horizon 5).

### 6. Re-run without re-downloading

Reuse cached market data and feature matrices:

```bash
vip run --symbol SPY --with vix --skip-ingest --skip-features
vip screen-batch --symbols SPY,QQQ --skip-ingest --skip-features
```

### 7. Optional: serve artifacts over HTTP

```bash
pip install -e ".[api]"
uvicorn vip.api.app:app --reload
```

Then open:

- `http://127.0.0.1:8000/experiments/` — list experiment IDs
- `http://127.0.0.1:8000/experiments/{id}` — `metrics.json` as JSON
- `http://127.0.0.1:8000/experiments/{id}/report` — HTML report

Override the artifacts root with `VIP_ARTIFACTS_ROOT` if needed. This server is
for local demos only (no authentication).

## Configuration

Defaults live in [`configs/default.yaml`](configs/default.yaml):

- Default symbol and date range
- Paths for raw / processed / artifact data
- Target horizon and RV estimator
- Primary and secondary evaluation metrics
- Logging level

CLI flags override config for a single run (for example `--symbol`, `--start`,
`--end`, `--n-splits`, `--embargo`).

## Project layout

```text
configs/           # YAML defaults
docs/              # Architecture, methodology, contributing
src/vip/           # Installable package
  application/     # Use-cases (ingest, features, screen, run study)
  cli/             # Typer commands
  domain/          # Entities, protocols, errors
  features/        # Targets and feature builders
  modeling/        # Baselines and regularized / tree models
  evaluation/      # Metrics, walk-forward, importance
  ingestion/       # yfinance adapter + validation
  persistence/     # Parquet + artifact stores
  reporting/       # HTML memos
  api/             # Optional FastAPI app
tests/             # Unit + integration (golden-file) tests
```

See [`docs/architecture.md`](docs/architecture.md) for layering and data flow.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Further reading:

- [`docs/contributing.md`](docs/contributing.md) — local setup and conventions
- [`docs/how_to_add_feature.md`](docs/how_to_add_feature.md) — add a factor end-to-end
- [`docs/research_methodology.md`](docs/research_methodology.md) — research design and caveats
- [`docs/README.md`](docs/README.md) — documentation index

## Built with

Developed with [Cursor](https://cursor.com) as an AI-assisted coding environment,
alongside standard Python tooling (pytest, Ruff, Typer, scikit-learn, etc.).
Architecture, research design, and review remain human-owned.

## License

MIT. See [LICENSE](LICENSE).