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
from pathlib import Path

import pandas as pd

from vip.domain.errors import PersistenceError
from vip.domain.value_objects import ExperimentId, Symbol
from vip.evaluation.comparison import summarize_walk_forward
from vip.evaluation.importance import (
    ImportanceOptions,
    WalkForwardSpec,
    permutation_importance_folds,
)
from vip.evaluation.shap_importance import shap_available, shap_importance_folds
from vip.modeling.tree_models import RandomForestVolModel
from vip.evaluation.stability import StabilityOptions, summarize_importance
from vip.evaluation.regimes import score_predictions_by_regime
from vip.evaluation.walk_forward import (
    collect_walk_forward_predictions, run_walk_forward
)
from vip.modeling.registry import create_default_model_registry
from vip.modeling.regularization import RidgeModel
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore
from vip.visualization.importance_plots import (
    plot_importance_bars, ImportancePlotOptions
)
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
class ScreenRunIdentity:
    """Identity fields for a factor-screen run.

    Parameters
    ----------
    symbol : Symbol
        Evaluated instrument.
    experiment_id : ExperimentId
        Artifact namespace for this run.
    screening_model : str
        Model used for permutation importance.

    Methods
    -------
    describe()
        Return a short human-readable summary.
    meta_payload()
        Return JSON-friendly identity fields for ``screen_meta``.
    """

    symbol: Symbol
    experiment_id: ExperimentId
    screening_model: str

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact identity summary.
        """
        return f"{self.symbol.value} / {self.experiment_id.value}"

    def meta_payload(self) -> dict[str, str]:
        """Return JSON-friendly identity fields.

        Returns
        -------
        dict of str to str
            Symbol and screening model for artifact metadata.
        """
        return {
            "symbol": self.symbol.value,
            "screening_model": self.screening_model,
        }

@dataclass(frozen=True, slots=True)
class ShapScreenOutputs:
    """Optional TreeSHAP outputs for the screening model path.

    Parameters
    ----------
    importance : pandas.DataFrame
        Fold-wise mean |SHAP| table.
    ranking : pandas.DataFrame
        Median-aggregated SHAP ranking.

    Methods
    -------
    top_feature()
        Return the top SHAP-ranked feature.
    describe()
        Return a short human-readable summary.
    """

    importance: pd.DataFrame
    ranking: pd.DataFrame

    def top_feature(self) -> str:
        """Return the top SHAP-ranked feature.

        Returns
        -------
        str
            Feature at the top of ``ranking``.
        """
        return str(self.ranking.iloc[0]["feature"])

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact SHAP summary.
        """
        return f"shap_top={self.top_feature()}"


@dataclass(frozen=True, slots=True)
class ScreenResultTables:
    """Tabular outputs from a factor-screen run.

    Parameters
    ----------
    summary : pandas.DataFrame
        Aggregate model horse-race metrics.
    fold_metrics : pandas.DataFrame
        Per-fold model metrics.
    importance : pandas.DataFrame
        Fold-wise permutation importance.
    ranking : pandas.DataFrame
        Aggregated factor ranking with stability stats.
    regime_metrics : pandas.DataFrame
        Regime-sliced OOS metrics.

    Methods
    -------
    top_feature()
        Return the highest-ranked feature name.
    best_model_qlike()
        Return the best horse-race model's mean QLIKE.
    """

    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    importance: pd.DataFrame
    ranking: pd.DataFrame
    regime_metrics: pd.DataFrame

    def top_feature(self) -> str:
        """Return the highest-ranked feature name.

        Returns
        -------
        str
            Feature at the top of ``ranking``.
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


@dataclass(frozen=True, slots=True)
class FactorScreenResult:
    """Summary of a completed factor-screening experiment.

    Parameters
    ----------
    identity : ScreenRunIdentity
        Symbol, experiment id, and screening model.
    tables : ScreenResultTables
        Horse-race, importance, ranking, and regime tables.

    Methods
    -------
    top_feature()
        Return the highest-ranked feature name.
    best_model_qlike()
        Return the best horse-race model's mean QLIKE.
    """

    identity: ScreenRunIdentity
    tables: ScreenResultTables
    shap: ShapScreenOutputs | None = None

    def top_feature(self) -> str:
        """Return the highest-ranked feature name.

        Returns
        -------
        str
            Feature at the top of the ranking table.
        """
        return self.tables.top_feature()

    def best_model_qlike(self) -> float:
        """Return the best horse-race model's mean QLIKE.

        Returns
        -------
        float
            Best model QLIKE from the horse-race summary.
        """
        return self.tables.best_model_qlike()


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
    regime_metrics = _run_regime_metrics(features, target, resolved)
    shap_outputs = _run_shap_importance(features, target, resolved)

    experiment_id = ExperimentId(
        f"factor-screen-{symbol.as_path_key().lower()}-{date.today().isoformat()}"
    )
    result = FactorScreenResult(
        identity=ScreenRunIdentity(
            symbol=symbol,
            experiment_id=experiment_id,
            screening_model=SCREENING_MODEL_NAME,
        ),
        tables=ScreenResultTables(
            summary=summary,
            fold_metrics=fold_metrics,
            importance=importance,
            ranking=ranking,
            regime_metrics=regime_metrics,
        ),
        shap=shap_outputs,
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
        options=StabilityOptions(top_k=config.top_k, rank_by="median"),
    )
    return importance, ranking


def _run_shap_importance(
    features: pd.DataFrame,
    target: pd.Series,
    config: ScreenConfig,
) -> ShapScreenOutputs | None:
    """Compute RF TreeSHAP importance when ``shap`` is installed.

    Parameters
    ----------
    features : pandas.DataFrame
        Predictor matrix.
    target : pandas.Series
        Realized-volatility target.
    config : ScreenConfig
        Walk-forward / ranking settings.

    Returns
    -------
    ShapScreenOutputs or None
        Fold-wise SHAP table and median ranking, or ``None`` if ``shap``
        is not installed.
    """
    if not shap_available():
        return None

    importance = shap_importance_folds(
        features=features,
        target=target,
        model_factory=RandomForestVolModel,
        fold_spec=WalkForwardSpec(
            n_splits=config.n_splits,
            embargo_size=config.embargo_size,
        ),
    )
    ranking = summarize_importance(
        importance,
        options=StabilityOptions(top_k=config.top_k, rank_by="median"),
    )
    return ShapScreenOutputs(importance=importance, ranking=ranking)


def _persist_screen_artifacts(
    artifact_store: FilesystemArtifactStore,
    result: FactorScreenResult,
    config: ScreenConfig,
) -> None:
    """Write screen artifacts (JSON, plots, HTML report)."""
    tables = result.tables
    identity = result.identity
    experiment_id = identity.experiment_id

    _write_screen_json_artifacts(
        artifact_store=artifact_store,
        experiment_id=experiment_id,
        identity=identity,
        tables=tables,
        top_feature=result.top_feature(),
    )

    _write_screen_plots(
        artifact_store=artifact_store,
        experiment_id=experiment_id,
        tables=tables,
        shap_outputs=result.shap,
    )

    _write_screen_html_report(
        artifact_store=artifact_store,
        experiment_id=experiment_id,
        config=config,
        identity=identity,
        tables=tables,
    )


def _write_screen_json_artifacts(
    artifact_store: FilesystemArtifactStore,
    experiment_id: ExperimentId,
    identity: ScreenRunIdentity,
    tables: ScreenResultTables,
    top_feature: str,
) -> None:
    """Write JSON artifacts for a factor-screen run."""
    artifact_store.write_json(
        experiment_id,
        "metrics",
        tables.summary.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "folds",
        tables.fold_metrics.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "importance",
        tables.importance.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "factor_ranking",
        tables.ranking.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "metrics_by_regime",
        tables.regime_metrics.to_dict(orient="records"),
    )
    artifact_store.write_json(
        experiment_id,
        "screen_meta",
        {
            **identity.meta_payload(),
            "top_feature": top_feature,
        },
    )


def _write_screen_plots(
    artifact_store: FilesystemArtifactStore,
    experiment_id: ExperimentId,
    tables: ScreenResultTables,
    shap_outputs: ShapScreenOutputs | None,
) -> Path:
    """Write plot PNGs and return permutation importance plot path."""
    plot_path = artifact_store.experiment_dir(experiment_id) / "importance_plot.png"
    plot_importance_bars(tables.ranking, plot_path)

    if shap_outputs is not None:
        shap_plot_path = (
            artifact_store.experiment_dir(experiment_id) / "shap_importance_plot.png"
        )
        plot_importance_bars(
            shap_outputs.ranking,
            shap_plot_path,
            options=ImportancePlotOptions(title="TreeSHAP importance (median/mean |SHAP|)"),
        )
        artifact_store.write_json(
            experiment_id,
            "shap_importance",
            shap_outputs.importance.to_dict(orient="records"),
        )
        artifact_store.write_json(
            experiment_id,
            "shap_ranking",
            shap_outputs.ranking.to_dict(orient="records"),
        )

    return plot_path


def _write_screen_html_report(
    artifact_store: FilesystemArtifactStore,
    experiment_id: ExperimentId,
    config: ScreenConfig,
    identity: ScreenRunIdentity,
    tables: ScreenResultTables,
) -> None:
    """Build and write the factor-screen HTML report."""
    plot_path = (
        artifact_store.experiment_dir(experiment_id) / "importance_plot.png"
    )
    payload = ScreenReportPayload(
        symbol=identity.symbol.value,
        experiment_id=experiment_id.value,
        screening_model=identity.screening_model,
        summary=tables.summary,
        ranking=tables.ranking,
        regime_metrics=tables.regime_metrics,
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


def _run_regime_metrics(
    features: pd.DataFrame,
    target: pd.Series,
    config: ScreenConfig,
) -> pd.DataFrame:
    """Collect OOS predictions and score locked regimes."""
    registry = create_default_model_registry()
    models = registry.create_many(list(HORSE_RACE_MODELS))
    predictions = collect_walk_forward_predictions(
        features=features,
        target=target,
        models=models,
        n_splits=config.n_splits,
        embargo_size=config.embargo_size,
    )
    return score_predictions_by_regime(predictions)
