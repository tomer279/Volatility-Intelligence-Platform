"""Unit tests for the optional FastAPI experiment artifact API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
from fastapi.testclient import TestClient

from vip.api.app import ARTIFACTS_ROOT_ENV_VAR, app


def _write_experiment(
    artifacts_dir: Path,
    experiment_id: str,
    metrics_payload: object,
    report_html: str,
) -> None:
    """Write minimal artifacts needed by the API.

    Parameters
    ----------
    artifacts_dir : pathlib.Path
        Root artifacts directory.
    experiment_id : str
        Experiment directory name.
    metrics_payload : object
        JSON-serializable payload written to ``metrics.json``.
    report_html : str
        HTML written to ``report.html``.
    """
    exp_dir = artifacts_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics_payload),
        encoding="utf-8",
    )
    (exp_dir / "report.html").write_text(
        report_html,
        encoding="utf-8",
    )


def test_api_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """API should list ids, return metrics JSON, and return report HTML."""
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    exp1 = "exp-001"
    exp2 = "exp-002"

    metrics_1 = [{"model": "ridge", "qlike": 0.1}]
    html_1 = "<html><body>Factor Screen</body></html>"

    metrics_2 = [{"model": "lasso", "qlike": 0.2}]
    html_2 = "<html><body>Factor Screen 2</body></html>"

    _write_experiment(artifacts_root, exp1, metrics_1, html_1)
    _write_experiment(artifacts_root, exp2, metrics_2, html_2)

    monkeypatch.setenv(ARTIFACTS_ROOT_ENV_VAR, str(artifacts_root))

    client = TestClient(app)

    resp = client.get("/experiments/")
    assert resp.status_code == 200
    assert resp.json() == [exp1, exp2]

    resp = client.get(f"/experiments/{exp1}")
    assert resp.status_code == 200
    assert resp.json() == metrics_1

    resp = client.get(f"/experiments/{exp1}/report")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "Factor Screen" in resp.text

    resp = client.get("/experiments/does-not-exist")
    assert resp.status_code == 404
