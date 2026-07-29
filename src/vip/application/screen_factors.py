"""Application use-case for factor screening experiments.

Exports
-------
ScreenConfig
    Walk-forward and importance settings for a screen run.
FactorScreenResult
    Summary of a completed factor-screening experiment.
screen_factors
    Load features, race models, rank factors, and persist artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import ExperimentId, Symbol
from vip.evaluation.comparison import summarize_walk_forward
from vip.evaluation.importance import (
    ImportanceOptions,
    WalkForwardSpec,
    permutation_importance_folds,
)
from vip.evaluation.stability import StabilityOptions, summarize_importance
from vip.evaluation.walk_forward import run_walk_forward
from vip.modeling.registry import create_default_model_registry
from vip.modeling.regularization import RidgeModel
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.visualization.importance_plots import plot_importance_bars
from vip.reporting.experiment_summary import (
    ReportMeta,
    ScreenReportPayload,
    build_factor_screen_context,
)
from vip.reporting.html_report import render_factor_screen_report, write_html_report

DEFAULT_TARGET_COLUMN = "target_rv_cc_5d"
DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO_SIZE = 5
DEFAULT_N_REPEATS = 5
DEFAULT_TOP_K = 3
DEFAULT_RANDOM_SEED = 0
SCREENING_MODEL_NAME = "ridge"
HORSE_RACE_MODELS = ("har_rv_ols", "ridge", "lasso")


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    """Settings for a factor-screening experiment.

    Parameters
    ----------
    n_splits : int, default 5
        Number of expanding walk-forward folds.
    embargo_size : int, default 5
        Embargo length in trading sessions.
    n_repeats : int, default 5
        Permutation repeats per feature per fold.
    top_k : int, default 3
        Top-k used for hit-rate stability.
    random_seed : int, default 0
        Base RNG seed for column shuffles.

    Methods
    -------
    validate()
        Raise if settings are invalid.
    describe()
        Return a short human-readable summary.
    """

    n_splits: int = DEFAULT_N_SPLITS
    embargo_size: int = DEFAULT_EMBARGO_SIZE
    n_repeats: int = DEFAULT_N_REPEATS
    top_k: int = DEFAULT_TOP_K
    random_seed: int = DEFAULT_RANDOM_SEED

    def validate(self) -> None:
        """Raise if any screen setting is invalid."""
        WalkForwardSpec(
            n_splits=self.n_splits,
            embargo_size=self.embargo_size,
        ).validate()
        ImportanceOptions(
            n_repeats=self.n_repeats,
            random_seed=self.random_seed,
        ).validate()
        StabilityOptions(top_k=self.top_k).validate()

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact description of the screen settings.
        """
        return (
            f"n_splits={self.n_splits}, embargo={self.embargo_size}, "
            f"n_repeats={self.n_repeats}, top_k={self.top_k}"
        )


@dataclass(frozen=True, slots=True)
class FactorScreenResult:
    """Summary of a completed factor-screening experiment.

    Parameters
    ----------
    symbol : Symbol
        Evaluated instrument.
    experiment_id : ExperimentId
        Artifact namespace for this run.
    summary : pandas.DataFrame
        Aggregate model horse-race metrics.
    fold_metrics : pandas.DataFrame
        Per-fold model metrics.
    importance : pandas.DataFrame
        Fold-wise Ridge permutation importance.
    ranking : pandas.DataFrame
        Aggregated factor ranking with stability stats.
    screening_model : str
        Model used for permutation importance.

    Methods
    -------
    top_feature()
        Return the highest-ranked feature name.
    best_model_qlike()
        Return the best horse-race model's mean QLIKE.
    """

    symbol: Symbol
    experiment_id: ExperimentId
    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    importance: pd.DataFrame
    ranking: pd.DataFrame
    screening_model: str

    def top_feature(self) -> str:
        """Return the highest-ranked feature name.

        Returns
        -------
        str
            Feature with largest mean permutation importance.
        """
        return str(self.ranking.iloc[0]["feature"])

    def best_model_qlike(self) -> float:
        """Return the best horse-race model's mean QLIKE.

        Returns
        -------
        float
            Mean QLIKE for the first row of ``summary``.
        """
        return float(self.summary.iloc[0]["qlike"])


def screen_factors(
    feature_store: ParquetFeatureMatrixStore,
    artifact_store: FilesystemArtifactStore,
    symbol: Symbol,
    config: ScreenConfig | None = None,
) -> FactorScreenResult:
    """Load features, race models, rank factors, and persist artifacts.

    Parameters
    ----------
    feature_store : ParquetFeatureMatrixStore
        Store containing the feature matrix.
    artifact_store : FilesystemArtifactStore
        Destination for JSON artifacts.
    symbol : Symbol
        Instrument to screen.
    config : ScreenConfig or None, default None
        Walk-forward / importance settings.

    Returns
    -------
    FactorScreenResult
        Horse-race tables, factor ranking, and experiment id.

    Raises
    ------
    PersistenceError
        If the feature matrix is missing or malformed.
    """
    resolved = config if config is not None else ScreenConfig()
    resolved.validate()

    features, target = _load_features_and_target(feature_store, symbol)
    fold_metrics, summary = _run_horse_race(features, target, resolved)
    importance, ranking = _run_ridge_importance(features, target, resolved)

    experiment_id = ExperimentId(
        f"factor-screen-{symbol.as_path_key().lower()}-{date.today().isoformat()}"
    )
    result = FactorScreenResult(
        symbol=symbol,
        experiment_id=experiment_id,
        summary=summary,
        fold_metrics=fold_metrics,
        importance=importance,
        ranking=ranking,
        screening_model=SCREENING_MODEL_NAME,
    )
    _persist_screen_artifacts(artifact_store, result, resolved)
    return result


def _load_features_and_target(
    feature_store: ParquetFeatureMatrixStore,
    symbol: Symbol,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and split the persisted feature matrix."""
    if not feature_store.exists(symbol):
        raise PersistenceError(
            f"No feature matrix found for {symbol.value}. Run features first."
        )
    matrix = feature_store.load(symbol)
    if DEFAULT_TARGET_COLUMN not in matrix.columns:
        raise PersistenceError(
            f"Feature matrix missing target column '{DEFAULT_TARGET_COLUMN}'."
        )
    target = matrix[DEFAULT_TARGET_COLUMN]
    features = matrix.drop(columns=[DEFAULT_TARGET_COLUMN])
    return features, target


def _run_horse_race(
    features: pd.DataFrame,
    target: pd.Series,
    config: ScreenConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate HAR OLS and regularized models walk-forward."""
    registry = create_default_model_registry()
    models = registry.create_many(list(HORSE_RACE_MODELS))
    fold_metrics = run_walk_forward(
        features=features,
        target=target,
        models=models,
        n_splits=config.n_splits,
        embargo_size=config.embargo_size,
    )
    summary = summarize_walk_forward(fold_metrics, primary_metric="qlike")
    return fold_metrics, summary


def _run_ridge_importance(
    features: pd.DataFrame,
    target: pd.Series,
    config: ScreenConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Ridge permutation importance and stability ranking."""
    importance = permutation_importance_folds(
        features=features,
        target=target,
        model_factory=RidgeModel,
        fold_spec=WalkForwardSpec(
            n_splits=config.n_splits,
            embargo_size=config.embargo_size,
        ),
        options=ImportanceOptions(
            n_repeats=config.n_repeats,
            random_seed=config.random_seed,
        ),
    )
    ranking = summarize_importance(
        importance,
        options=StabilityOptions(top_k=config.top_k),
    )
    return importance, ranking


def _persist_screen_artifacts(
    artifact_store: FilesystemArtifactStore,
    result: FactorScreenResult,
    config: ScreenConfig,
) -> None:
    """Write metrics, importance, and ranking JSON artifacts."""
    experiment_id = result.experiment_id
    artifact_store.write_json(
        experiment_id,
        "metrics",
        result.summary.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "folds",
        result.fold_metrics.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "importance",
        result.importance.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "factor_ranking",
        result.ranking.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "screen_meta",
        {
            "symbol": result.symbol.value,
            "screening_model": result.screening_model,
            "top_feature": result.top_feature(),
        },
    )
    plot_path = artifact_store.experiment_dir(experiment_id) / "importance_plot.png"
    plot_importance_bars(result.ranking, plot_path)
    payload = ScreenReportPayload(
        symbol=result.symbol.value,
        experiment_id=result.experiment_id.value,
        screening_model=result.screening_model,
        summary=result.summary,
        ranking=result.ranking,
    )
    meta = ReportMeta(
        n_splits=config.n_splits,
        embargo_size=config.embargo_size,
    )
    context = build_factor_screen_context(payload, plot_path, meta)
    html = render_factor_screen_report(context)
    write_html_report(
        artifact_store.experiment_dir(experiment_id) / "report.html",
        html,
    )
