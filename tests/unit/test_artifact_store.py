"""Tests for filesystem artifact storage."""

from pathlib import Path

import pytest

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import ExperimentId
from vip.persistence import FilesystemArtifactStore


def test_json_artifact_roundtrip(tmp_path: Path) -> None:
    """JSON artifacts round-trip through the store."""
    store = FilesystemArtifactStore(tmp_path)
    experiment_id = ExperimentId("exp-001")
    payload = {"primary_metric": "qlike", "score": 0.12}

    written = store.write_json(experiment_id, "metrics", payload)
    assert written.is_file()
    assert store.read_json(experiment_id, "metrics") == payload


def test_read_missing_artifact_raises(tmp_path: Path) -> None:
    """Reading a missing artifact raises PersistenceError."""
    store = FilesystemArtifactStore(tmp_path)
    with pytest.raises(PersistenceError):
        store.read_json(ExperimentId("exp-001"), "metrics")