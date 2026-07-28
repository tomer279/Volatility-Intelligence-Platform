"""Filesystem storage for experiment artifacts.

Exports
-------
FilesystemArtifactStore
    Read/write JSON artifacts under an experiment directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import ExperimentId


class FilesystemArtifactStore:
    """Store experiment artifacts as files on disk.

    Parameters
    ----------
    root_dir : pathlib.Path
        Root artifacts directory (for example ``data/artifacts``).

    Methods
    -------
    experiment_dir(experiment_id)
        Return the directory for an experiment.
    write_json(experiment_id, name, payload)
        Persist a JSON artifact.
    read_json(experiment_id, name)
        Load a JSON artifact.
    """

    def __init__(self, root_dir: Path) -> None:
        """Initialize the store.

        Parameters
        ----------
        root_dir : pathlib.Path
            Root directory for all experiment outputs.
        """
        self._root_dir = root_dir

    def experiment_dir(self, experiment_id: ExperimentId) -> Path:
        """Return the directory for an experiment.

        Parameters
        ----------
        experiment_id : ExperimentId
            Experiment namespace.

        Returns
        -------
        pathlib.Path
            Path of the form ``{root}/{experiment_id}``.
        """
        return self._root_dir / experiment_id.as_path_key()

    def write_json(
        self,
        experiment_id: ExperimentId,
        name: str,
        payload: dict[str, Any],
    ) -> Path:
        """Persist a JSON-serializable artifact.

        Parameters
        ----------
        experiment_id : ExperimentId
            Experiment namespace.
        name : str
            Artifact basename without extension.
        payload : dict of str to Any
            JSON-serializable mapping.

        Returns
        -------
        pathlib.Path
            Path written to disk.

        Raises
        ------
        PersistenceError
            If the artifact cannot be written.
        """
        directory = self.experiment_dir(experiment_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            raise PersistenceError(
                f"Failed to write artifact '{name}' for "
                f"{experiment_id.value}: {exc}"
            ) from exc
        return path

    def read_json(
        self,
        experiment_id: ExperimentId,
        name: str,
    ) -> dict[str, Any]:
        """Load a JSON artifact.

        Parameters
        ----------
        experiment_id : ExperimentId
            Experiment namespace.
        name : str
            Artifact basename without extension.

        Returns
        -------
        dict of str to Any
            Loaded payload.

        Raises
        ------
        PersistenceError
            If the artifact is missing or invalid JSON.
        """
        path = self.experiment_dir(experiment_id) / f"{name}.json"
        if not path.is_file():
            raise PersistenceError(
                f"Artifact '{name}' not found for {experiment_id.value}"
            )
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PersistenceError(
                f"Failed to read artifact '{name}' for "
                f"{experiment_id.value}: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise PersistenceError(
                f"Artifact '{name}' for {experiment_id.value} "
                "must be a JSON object"
            )
        return loaded
