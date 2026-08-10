"""Application use-case for factor screening experiments.

Delegates the model horse-race to ``screen_horse_race`` and artifact
persistence to ``screen_factor_artifacts``.

Exports
-------
ScreenConfig
    Walk-forward and importance settings for a screen run.
ScreenInferenceOptions
    Bootstrap / HLN settings for horse-race inference vs HAR.
target_column_for_horizon
    Name ``target_rv_cc_{h}d`` for a forecast horizon.
settings_for_horizon
    Build ``ScreenConfig`` + ``ScreenInferenceOptions`` with M8 defaults.
ScreenArtifactContext
    Persist-time screen + inference settings for artifacts.
FactorScreenResult
    Summary of a completed factor-screening experiment.
screen_factors
    Load features, race models, rank factors, run inference, and persist artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from vip.application.screen_horse_race import (
    HorseRaceOptions,
    run_horse_race_with_inference,
)
from vip.application.screen_factor_artifacts import persist_screen_artifacts
from vip.domain.errors import DataValidationError, PersistenceError
from vip.domain.value_objects import ExperimentId, Symbol
from vip.evaluation.comparison import InferenceSummaryOptions
from vip.evaluation.horizon_defaults import (
    allowed_bootstrap_block_range,
    default_bootstrap_block_length,
    default_embargo_for_horizon,
)
from vip.evaluation.inference import (
    BootstrapBlockBounds,
    BootstrapInferenceOptions,
    nw_lags_for_horizon,
)
from vip.features.targets import TARGET_NAME_PREFIX
from vip.evaluation.importance import (
    ImportanceOptions,
    WalkForwardSpec,
    permutation_importance_folds,
    DEFAULT_IMPORTANCE_DELTA_CAP
)
from vip.evaluation.shap_importance import shap_available, shap_importance_folds
from vip.modeling.tree_models import RandomForestVolModel
from vip.evaluation.stability import StabilityOptions, summarize_importance
from vip.evaluation.regimes import score_predictions_by_regime

from vip.modeling.regularization import RidgeModel
from vip.persistence.artifact_store import FilesystemArtifactStore
from vip.persistence.feature_matrix_store import ParquetFeatureMatrixStore


DEFAULT_BASELINE_MODEL = "har_rv_ols"
DEFAULT_HORIZON_DAYS = 5
DEFAULT_INCLUDE_HLN_DM = True
DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO_SIZE = 5
DEFAULT_N_REPEATS = 5
DEFAULT_TOP_K = 3
DEFAULT_RANDOM_SEED = 0
SCREENING_MODEL_NAME = "ridge"
DEFAULT_INCLUDE_NONOVERLAP_SENSITIVITY = True
_MIN_HORIZON_DAYS = 1


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    """Settings for a factor-screening experiment.

    Parameters
    ----------
    n_splits : int, default 5
        Number of expanding walk-forward folds.
    embargo_size : int, default 5
        Embargo length in trading sessions. For multi-horizon runs set
        ``embargo_size = horizon_days`` (see ``settings_for_horizon`` /
        ``default_embargo_for_horizon``).
    n_repeats : int, default 5
        Permutation repeats per feature per fold.
    top_k : int, default 3
        Top-k used for hit-rate stability.
    random_seed : int, default 0
        Base RNG seed for column shuffles.
    importance_delta_cap : float or None, default 1.0
        Clip per-shuffle ΔQLIKE to ``[-cap, cap]`` before fold aggregation.
        ``None`` disables clipping (diagnostics only).

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
    importance_delta_cap: float | None = DEFAULT_IMPORTANCE_DELTA_CAP

    def validate(self) -> None:
        """Raise if any screen setting is invalid."""
        WalkForwardSpec(
            n_splits=self.n_splits,
            embargo_size=self.embargo_size,
        ).validate()
        ImportanceOptions(
            n_repeats=self.n_repeats,
            random_seed=self.random_seed,
            delta_cap=self.importance_delta_cap,
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
            f"n_repeats={self.n_repeats}, top_k={self.top_k}, "
            f"importance_delta_cap={self.importance_delta_cap}"
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
        Horse-race metrics plus bootstrap / optional HLN–DM vs baseline.
    fold_metrics : pandas.DataFrame
        Per-fold model metrics.
    importance : pandas.DataFrame
        Fold-wise permutation importance.
    ranking : pandas.DataFrame
        Aggregated factor ranking with stability stats.
    regime_metrics : pandas.DataFrame
        Regime-sliced OOS metrics.
    oos_losses : pandas.DataFrame
        Per-row OOS QLIKE loss panel indexed by session date.

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
    oos_losses: pd.DataFrame

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
class ScreenInferenceOptions:
    """Inference settings for a factor-screen horse-race.

    Parameters
    ----------
    baseline_model : str
        Reference model for ΔQLIKE.
    horizon_days : int
        Target horizon (NW lags = horizon − 1).
    include_hln_dm : bool
        Attach secondary HLN–DM columns when True.
    bootstrap : BootstrapInferenceOptions
        Primary block-bootstrap settings.
    include_nonoverlap_sensitivity : bool
        When True, persist ``inference_sensitivity.json`` footnote bootstrap.

    Methods
    -------
    validate()
        Raise if settings are invalid.
    to_summary_options()
        Convert to ``InferenceSummaryOptions``.
    """

    baseline_model: str = DEFAULT_BASELINE_MODEL
    horizon_days: int = DEFAULT_HORIZON_DAYS
    include_hln_dm: bool = DEFAULT_INCLUDE_HLN_DM
    bootstrap: BootstrapInferenceOptions = field(
        default_factory=BootstrapInferenceOptions
    )
    include_nonoverlap_sensitivity: bool = DEFAULT_INCLUDE_NONOVERLAP_SENSITIVITY


    def validate(self) -> None:
        """Raise if inference settings are invalid."""
        self.to_summary_options().validate()

    def to_summary_options(self) -> InferenceSummaryOptions:
        """Convert to comparison-layer options.

        Returns
        -------
        InferenceSummaryOptions
            Options consumed by ``summarize_with_inference``.
        """
        return InferenceSummaryOptions(
            baseline_model=self.baseline_model,
            horizon_days=self.horizon_days,
            include_hln_dm=self.include_hln_dm,
            bootstrap=self.bootstrap,
        )


@dataclass(frozen=True, slots=True)
class FactorScreenResult:
    """Summary of a completed factor-screening experiment.

    Parameters
    ----------
    identity : ScreenRunIdentity
        Symbol, experiment id, and screening model.
    tables : ScreenResultTables
        Inference-enriched horse-race, OOS losses, importance, ranking,
        and regime tables.
    shap : ShapScreenOutputs or None
        Optional TreeSHAP outputs when ``shap`` is installed.

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


@dataclass(frozen=True, slots=True)
class ScreenArtifactContext:
    """Persist-time settings for a factor-screen run.

    Parameters
    ----------
    config : ScreenConfig
        Walk-forward / importance settings.
    inference : ScreenInferenceOptions
        Bootstrap / HLN settings used for artifacts and meta.

    Methods
    -------
    describe()
        Return a short human-readable summary.
    screen_meta_payload(identity, top_feature)
        Build the ``screen_meta.json`` mapping.
    """

    config: ScreenConfig
    inference: ScreenInferenceOptions

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact persist-context summary.
        """
        return f"{self.config.describe()} | {self.inference.to_summary_options().describe()}"

    def screen_meta_payload(
        self,
        identity: ScreenRunIdentity,
        top_feature: str,
    ) -> dict[str, object]:
        """Build JSON fields for ``screen_meta``.

        Parameters
        ----------
        identity : ScreenRunIdentity
            Symbol / screening-model identity.
        top_feature : str
            Top-ranked factor name.

        Returns
        -------
        dict of str to object
            Meta payload including inference defaults.
        """
        inference = self.inference
        return {
            **identity.meta_payload(),
            "top_feature": top_feature,
            "baseline_model": inference.baseline_model,
            "horizon_days": inference.horizon_days,
            "target_column": target_column_for_horizon(inference.horizon_days),
            "embargo_size": self.config.embargo_size,
            "nw_lags": nw_lags_for_horizon(inference.horizon_days),
            "bootstrap_block_length": inference.bootstrap.block_length,
            "bootstrap_n_resamples": inference.bootstrap.n_resamples,
            "alpha": inference.bootstrap.alpha,
            "include_hln_dm": inference.include_hln_dm,
            "include_nonoverlap_sensitivity": inference.include_nonoverlap_sensitivity,
            "importance_delta_cap": self.config.importance_delta_cap,
        }


def target_column_for_horizon(horizon_days: int) -> str:
    """Return the locked close-to-close target column for a horizon.

    Parameters
    ----------
    horizon_days : int
        Forecast horizon in trading days (must be >= 1).

    Returns
    -------
    str
        Column name ``target_rv_cc_{horizon_days}d``.

    Raises
    ------
    DataValidationError
        If ``horizon_days`` is less than 1.
    """
    if horizon_days < _MIN_HORIZON_DAYS:
        raise DataValidationError("horizon_days must be at least 1.")
    return f"{TARGET_NAME_PREFIX}{horizon_days}d"


def settings_for_horizon(horizon_days: int) -> ScreenArtifactContext:
    """Build screen + inference settings for one forecast horizon.

    Uses Agent A helpers: ``embargo_size = horizon_days``, horizon-aware
    bootstrap ``block_length`` and ``block_bounds``. Defaults match the
    locked M8 table for ``h ∈ {1, 5, 21}``.

    Parameters
    ----------
    horizon_days : int
        Forecast horizon in trading days. Must be a locked screen horizon
        when using default bootstrap length (see
        ``default_bootstrap_block_length``).

    Returns
    -------
    ScreenArtifactContext
        Validated ``config`` + ``inference`` ready for ``screen_factors``.

    Raises
    ------
    DataValidationError
        If horizon or derived bootstrap settings are invalid.
    """
    embargo = default_embargo_for_horizon(horizon_days)
    block_length = default_bootstrap_block_length(horizon_days)
    low, high = allowed_bootstrap_block_range(horizon_days)
    config = ScreenConfig(embargo_size=embargo)
    inference = ScreenInferenceOptions(
        horizon_days=horizon_days,
        bootstrap=BootstrapInferenceOptions(
            block_length=block_length,
            block_bounds=BootstrapBlockBounds(
                minimum=low,
                maximum=high,
            ),
        ),
    )
    config.validate()
    inference.validate()
    return ScreenArtifactContext(config=config, inference=inference)


def screen_factors(
        feature_store: ParquetFeatureMatrixStore,
        artifact_store: FilesystemArtifactStore,
        symbol: Symbol,
        config: ScreenConfig | None = None,
        inference: ScreenInferenceOptions | None = None,
) -> FactorScreenResult:
    """Load features, race models, rank factors, run inference, and persist artifacts.

    Parameters
    ----------
    feature_store : ParquetFeatureMatrixStore
        Store containing the feature matrix.
    artifact_store : FilesystemArtifactStore
        Destination for JSON / plot / HTML artifacts.
    symbol : Symbol
        Instrument to screen.
    config : ScreenConfig or None, default None
        Walk-forward / importance settings.
    inference : ScreenInferenceOptions or None, default None
        Bootstrap / HLN settings vs the HAR baseline.

    Returns
    -------
    FactorScreenResult
        Horse-race tables (with inference columns), factor ranking,
        OOS losses, and experiment id.

    Raises
    ------
    PersistenceError
        If the feature matrix is missing or malformed.
    """
    context = _resolve_artifact_context(config, inference)
    target_column = target_column_for_horizon(context.inference.horizon_days)
    features, target = _load_features_and_target(
        feature_store, symbol, target_column
    )
    tables, shap_outputs = _build_screen_tables(
        features, target, context.config, context.inference
    )
    result = _assemble_screen_result(symbol, tables, shap_outputs)
    persist_screen_artifacts(artifact_store, result, context)
    return result


def _resolve_artifact_context(
    config: ScreenConfig | None,
    inference: ScreenInferenceOptions | None,
) -> ScreenArtifactContext:
    """Validate and bundle screen + inference settings."""
    resolved = config if config is not None else ScreenConfig()
    resolved.validate()
    inference_opts = (
        inference if inference is not None else ScreenInferenceOptions()
    )
    inference_opts.validate()
    return ScreenArtifactContext(config=resolved, inference=inference_opts)


def _build_screen_tables(
    features: pd.DataFrame,
    target: pd.Series,
    config: ScreenConfig,
    inference: ScreenInferenceOptions,
) -> tuple[ScreenResultTables, ShapScreenOutputs | None]:
    """Run horse-race, inference, importance, regimes, and optional SHAP."""
    fold_metrics, oos_losses, summary = run_horse_race_with_inference(
        features,
        target,
        HorseRaceOptions(
            n_splits=config.n_splits,
            embargo_size=config.embargo_size,
            summary_options=inference.to_summary_options(),
        ),
    )
    importance, ranking = _run_ridge_importance(features, target, config)
    regime_metrics = _run_regime_metrics_from_losses(oos_losses)
    shap_outputs = _run_shap_importance(features, target, config)
    tables = ScreenResultTables(
        summary=summary,
        fold_metrics=fold_metrics,
        importance=importance,
        ranking=ranking,
        regime_metrics=regime_metrics,
        oos_losses=oos_losses,
    )
    return tables, shap_outputs


def _assemble_screen_result(
    symbol: Symbol,
    tables: ScreenResultTables,
    shap_outputs: ShapScreenOutputs | None,
) -> FactorScreenResult:
    """Build the public result object with a dated experiment id."""
    experiment_id = ExperimentId(
        f"factor-screen-{symbol.as_path_key().lower()}-{date.today().isoformat()}"
    )
    identity = ScreenRunIdentity(
        symbol=symbol,
        experiment_id=experiment_id,
        screening_model=SCREENING_MODEL_NAME,
    )
    return FactorScreenResult(
        identity=identity,
        tables=tables,
        shap=shap_outputs,
    )


def _load_features_and_target(
        feature_store: ParquetFeatureMatrixStore,
        symbol: Symbol,
        target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and split the persisted feature matrix."""
    if not feature_store.exists(symbol):
        raise PersistenceError(
            f"No feature matrix found for {symbol.value}. Run features first."
        )
    matrix = feature_store.load(symbol)
    if target_column not in matrix.columns:
        raise PersistenceError(
            f"Feature matrix missing target column '{target_column}'."
        )
    target = matrix[target_column]
    features = matrix.drop(columns=[target_column])
    return features, target


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
            delta_cap=config.importance_delta_cap,
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


def _run_regime_metrics_from_losses(oos_losses: pd.DataFrame) -> pd.DataFrame:
    """Score locked regimes from the shared OOS prediction/loss panel.

    Parameters
    ----------
    oos_losses : pandas.DataFrame
        Output of ``attach_qlike_losses`` (must include ``model``,
        ``y_true``, ``y_pred``). Extra columns such as ``qlike_loss``
        are ignored.

    Returns
    -------
    pandas.DataFrame
        Regime × model QLIKE / MSE / MAE table.
    """
    return score_predictions_by_regime(oos_losses)
