# Contributing

## Setup

1. Create and activate a virtual environment from the repo root.

```bash
python -m venv .venv
```

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# If execution policy blocks Activate.ps1:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```bash
# macOS / Linux
source .venv/bin/activate
```

2. Install with dev extras:

```bash
pip install -e ".[dev]"
```

On Windows, if `python` is a Store stub, use `py -3.11 -m venv .venv` and `py -m pip install -e ".[dev]"`.

## Checks

```bash
vip --help
vip info
pytest -q
```

## Code style

- Docstrings: NumPy style (modules, classes, public functions).
- Prefer typed errors from `vip.domain.errors` over bare exceptions.
- Keep domain free of vendor SDKs (yfinance, etc.).
- New features/models should plug in via registries/protocols, not by editing orchestrators.

## Project layout

See [`architecture.md`](architecture.md) for layering and data flow, and the root [`README.md`](../README.md) for how to run the CLI. Folder-level READMEs under `src/vip/` describe each package. Milestone history lives in [`plan.md`](../plan.md) and [`milestones/`](milestones/) (optional reading).