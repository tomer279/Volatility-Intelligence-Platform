# Contributing

## Setup

1. Create and activate a virtual environment.
2. From the repo root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

If PowerShell blocks `Activate.ps1`, either use the full path above or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Checks

```powershell
vip --help
vip info
py -m pytest -q
```

Prefer `py` / the venv interpreter over the Windows `python` stub.

## Code style

- Docstrings: NumPy style (modules, classes, public functions).
- Prefer typed errors from `vip.domain.errors` over bare exceptions.
- Keep domain free of vendor SDKs (yfinance, etc.).
- New features/models should plug in via registries/protocols, not by editing orchestrators.

## Project layout

See `plan.md` for architecture and milestones. Folder-level READMEs under `src/vip/` describe each package.

---