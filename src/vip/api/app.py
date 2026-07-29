"""FastAPI app for serving experiment artifacts.

Exports
-------
app
    FastAPI application exposing:
    - GET ``/experiments/`` to list experiment IDs.
    - GET ``/experiments/{id}`` to return ``metrics.json`` as JSON.
    - GET ``/experiments/{id}/report`` to return ``report.html`` as HTML.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError as exc:
    raise ImportError(
        "vip.api requires optional dependencies. "
        "Install with: pip install -e '.[api]'."
    ) from exc

from vip.domain.errors import DataValidationError
from vip.domain.value_objects import ExperimentId
from vip.persistence.artifact_store import FilesystemArtifactStore

ARTIFACTS_ROOT_ENV_VAR = "VIP_ARTIFACTS_ROOT"
DEFAULT_ARTIFACTS_ROOT = Path("data") / "artifacts"

app = FastAPI(title="VIP Experiment Artifacts", version="0.1.0")


def _get_artifacts_root() -> Path:
    """Resolve the experiment artifacts root directory.

    Returns
    -------
    pathlib.Path
        Artifacts root directory (environment override or default).
    """
    env_value = os.environ.get(ARTIFACTS_ROOT_ENV_VAR)
    if env_value:
        return Path(env_value)
    return DEFAULT_ARTIFACTS_ROOT


def _artifact_store(artifacts_root: Path) -> FilesystemArtifactStore:
    """Create a filesystem artifact store for a given root.

    Parameters
    ----------
    artifacts_root : pathlib.Path
        Root directory containing per-experiment subdirectories.

    Returns
    -------
    vip.persistence.artifact_store.FilesystemArtifactStore
        Filesystem-backed artifact store.
    """
    return FilesystemArtifactStore(root_dir=artifacts_root)


def _list_experiment_ids(artifacts_root: Path) -> list[str]:
    """List experiment IDs that appear to have metrics available.

    Parameters
    ----------
    artifacts_root : pathlib.Path
        Artifacts root directory.

    Returns
    -------
    list[str]
        Sorted experiment directory names that include ``metrics.json``.
    """
    if not artifacts_root.is_dir():
        return []

    ids: list[str] = []
    for child in artifacts_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "metrics.json").is_file():
            ids.append(child.name)

    ids.sort()
    return ids


def _read_metrics_payload(
    store: FilesystemArtifactStore,
    experiment_id: ExperimentId,
) -> Any:
    """Read and parse ``metrics.json`` for an experiment.

    Parameters
    ----------
    store : FilesystemArtifactStore
        Artifact store used to locate the experiment directory.
    experiment_id : vip.domain.value_objects.ExperimentId
        Experiment identifier.

    Returns
    -------
    Any
        Parsed JSON payload from ``metrics.json`` (e.g., list[dict]).
    """
    metrics_path = store.experiment_dir(experiment_id) / "metrics.json"
    raw = metrics_path.read_text(encoding="utf-8")
    return json.loads(raw)


def _read_report_html(
    store: FilesystemArtifactStore,
    experiment_id: ExperimentId,
) -> str:
    """Read report HTML for an experiment.

    Parameters
    ----------
    store : FilesystemArtifactStore
        Artifact store used to locate the experiment directory.
    experiment_id : vip.domain.value_objects.ExperimentId
        Experiment identifier.

    Returns
    -------
    str
        Raw HTML string from ``report.html``.
    """
    report_path = store.experiment_dir(experiment_id) / "report.html"
    return report_path.read_text(encoding="utf-8")


@app.get("/experiments/", response_model=list[str])
def list_experiments() -> list[str]:
    """List available experiment IDs.

    Returns
    -------
    list[str]
        Experiment IDs found under the configured artifacts root.
    """
    artifacts_root = _get_artifacts_root()
    return _list_experiment_ids(artifacts_root)


@app.get("/experiments/{experiment_id}")
def get_experiment_metrics(experiment_id: str) -> Any:
    """Return ``metrics.json`` content as JSON.

    Parameters
    ----------
    id : str
        Experiment identifier.

    Returns
    -------
    Any
        Parsed JSON payload from ``metrics.json``.
    """
    try:
        experiment_id = ExperimentId(experiment_id)
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = _artifact_store(_get_artifacts_root())
    metrics_path = store.experiment_dir(experiment_id) / "metrics.json"
    if not metrics_path.is_file():
        raise HTTPException(status_code=404, detail="Experiment not found")

    try:
        return _read_metrics_payload(store, experiment_id)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Invalid metrics.json (not valid JSON)",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read metrics.json",
        ) from exc


@app.get("/experiments/{experiment_id}/report")
def get_experiment_report(experiment_id: str) -> HTMLResponse:
    """Return experiment HTML report.

    Parameters
    ----------
    id : str
        Experiment identifier.

    Returns
    -------
    fastapi.responses.HTMLResponse
        HTML content from ``report.html``.
    """
    try:
        experiment_id = ExperimentId(experiment_id)
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = _artifact_store(_get_artifacts_root())
    report_path = store.experiment_dir(experiment_id) / "report.html"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Experiment not found")

    try:
        html = _read_report_html(store, experiment_id)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not read report.html") from exc

    return HTMLResponse(content=html)
