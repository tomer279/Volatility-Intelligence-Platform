"""Application use-case for baseline walk-forward experiments.

Exports
-------
BaselineExperimentResult
    Summary of a completed baseline experiment.
run_baseline_experiment
    Load features, evaluate baselines, and persist metrics artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import ExperimentId, Symbol
from vip.evaluation.comparison import summarize_walk_forward
from vip.evaluation.walk_forward import run_walk_forward
from vip.modeling.baselines import EwmaModel, HarRvOlsModel, HistoricalMeanModel
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore

DEFAULT_TARGET_COLUMN = "target_rv_cc_5d"
DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO_SIZE = 5


@dataclass(frozen=True, slots=True)
class BaselineExperimentResult:
    """Summary of a completed baseline experiment.

    Parameters
    ----------
    symbol : Symbol
        Evaluated instrument.
    experiment_id : ExperimentId
        Artifact namespace for this run.
    summary : pandas.DataFrame
        Aggregate metrics by model.
    fold_metrics : pandas.DataFrame
        Per-fold metrics.
    primary_model : str
        Best model by QLIKE.

    Methods
    -------
    best_qlike()
        Return the winning model's mean QLIKE.
    """

    symbol: Symbol
    experiment_id: ExperimentId
    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    primary_model: str

    def best_qlike(self) -> float:
        """Return the winning model's mean QLIKE.

        Returns
        -------
        float
            Mean QLIKE for ``primary_model``.
        """
        row = self.summary.loc[self.summary["model"] == self.primary_model, "qlike"]
        return float(row.iloc[0])


def run_baseline_experiment(
    feature_store: ParquetFeatureMatrixStore,
    artifact_store: FilesystemArtifactStore,
    symbol: Symbol,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_size: int = DEFAULT_EMBARGO_SIZE,
) -> BaselineExperimentResult:
    """Load features, evaluate baselines, and persist metrics artifacts.

    Parameters
    ----------
    feature_store : ParquetFeatureMatrixStore
        Store containing the feature matrix.
    artifact_store : FilesystemArtifactStore
        Destination for metrics JSON artifacts.
    symbol : Symbol
        Instrument to evaluate.
    n_splits : int, default 5
        Number of walk-forward folds.
    embargo_size : int, default 5
        Embargo length in trading sessions.

    Returns
    -------
    BaselineExperimentResult
        Summary plus persisted artifact identifiers.

    Raises
    ------
    PersistenceError
        If the feature matrix is missing.
    """
    if not feature_store.exists(symbol):
        raise PersistenceError(
            f"No feature matrix found for {symbol.value}. Run features first."
        )

    matrix = feature_store.load(symbol)
    features, target = _split_matrix(matrix)
    models = {
        "historical_mean": HistoricalMeanModel(),
        "ewma": EwmaModel(),
        "har_rv_ols": HarRvOlsModel(),
    }

    fold_metrics = run_walk_forward(
        features=features,
        target=target,
        models=models,
        n_splits=n_splits,
        embargo_size=embargo_size,
    )
    summary = summarize_walk_forward(fold_metrics, primary_metric="qlike")
    primary_model = str(summary.iloc[0]["model"])

    experiment_id = ExperimentId(
        f"baselines-{symbol.as_path_key().lower()}-{date.today().isoformat()}"
    )
    _persist_artifacts(artifact_store, experiment_id, summary, fold_metrics)

    return BaselineExperimentResult(
        symbol=symbol,
        experiment_id=experiment_id,
        summary=summary,
        fold_metrics=fold_metrics,
        primary_model=primary_model,
    )


def _split_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature matrix into features and target.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Persisted feature matrix including the target column.

    Returns
    -------
    tuple of pandas.DataFrame and pandas.Series
        Feature frame and target series.

    Raises
    ------
    PersistenceError
        If the expected target column is missing.
    """
    if DEFAULT_TARGET_COLUMN not in matrix.columns:
        raise PersistenceError(
            f"Feature matrix missing target column '{DEFAULT_TARGET_COLUMN}'."
        )
    target = matrix[DEFAULT_TARGET_COLUMN]
    features = matrix.drop(columns=[DEFAULT_TARGET_COLUMN])
    return features, target


def _persist_artifacts(
    artifact_store: FilesystemArtifactStore,
    experiment_id: ExperimentId,
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    """Write summary and fold-metric JSON artifacts.

    Parameters
    ----------
    artifact_store : FilesystemArtifactStore
        Artifact destination.
    experiment_id : ExperimentId
        Experiment namespace.
    summary : pandas.DataFrame
        Aggregate metrics table.
    fold_metrics : pandas.DataFrame
        Per-fold metrics table.
    """
    artifact_store.write_json(
        experiment_id,
        "metrics",
        summary.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "folds",
        fold_metrics.to_dict(orient="records"),
    )