# `vip.api`

## Purpose
Optional FastAPI app that serves experiment artifacts from the local artifact store for demos.

## Modules
- `app.py` - FastAPI routes over `FilesystemArtifactStore`.
- `__init__.py` - Package docstring / exports.

## Key APIs
- `GET /experiments/` - List experiment IDs that have `metrics.json`.
- `GET /experiments/{id}` - Return `metrics.json` as JSON.
- `GET /experiments/{id}/report` - Return `report.html` as HTML.
- `VIP_ARTIFACTS_ROOT` - Optional env override for the artifacts directory (default `data/artifacts`).

## Dependencies
- Requires optional extras: `pip install -e ".[api]"` (`fastapi`, `uvicorn`).
- Depends on: `vip.persistence.artifact_store`, `vip.domain.value_objects`.

## Usage
```bash
pip install -e ".[api]"
uvicorn vip.api.app:app --reload
```

## Notes
- Local demo only; no authentication.
- Install fails clearly if FastAPI is missing (ImportError with install hint).
