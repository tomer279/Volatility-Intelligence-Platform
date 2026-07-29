"""Permutation importance for volatility models under QLIKE.

Exports
-------
WalkForwardSpec
    Expanding walk-forward settings for importance runs.
ImportanceOptions
    Repeat count and RNG seed for column shuffles.
permutation_importance_folds
    Fit per fold, then rank features by ΔQLIKE under test shuffles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vip.domain.errors import DataValidationError
from vip.evaluation.metrics import qlike
from vip.evaluation.splitting import WalkForwardFold, generate_expanding_folds
from vip.evaluation.walk_forward import _align_features_and_target

ModelFactory = Callable[[], Any]
DEFAULT_N_REPEATS = 5
DEFAULT_RANDOM_SEED = 0
DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO_SIZE = 5
FOLD_SEED_STRIDE = 1_000_003
FEATURE_SEED_STRIDE = 10_007


@dataclass(frozen=True, slots=True)
class WalkForwardSpec:
    """Expanding walk-forward settings shared by importance helpers.

    Parameters
    ----------
    n_splits : int, default 5
        Number of expanding test folds.
    embargo_size : int, default 5
        Embargo length in rows between train and test.

    Methods
    -------
    validate()
        Raise if settings are invalid.
    describe()
        Return a short human-readable summary.
    """

    n_splits: int = DEFAULT_N_SPLITS
    embargo_size: int = DEFAULT_EMBARGO_SIZE

    def validate(self) -> None:
        """Raise ``DataValidationError`` when settings are invalid."""
        if self.n_splits < 2:
            raise DataValidationError("n_splits must be at least 2.")
        if self.embargo_size < 0:
            raise DataValidationError("embargo_size must be non-negative.")

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact description of the fold settings.
        """
        return f"n_splits={self.n_splits}, embargo_size={self.embargo_size}"


@dataclass(frozen=True, slots=True)
class ImportanceOptions:
    """Options for permutation importance repeats.

    Parameters
    ----------
    n_repeats : int, default 5
        Number of shuffles per feature per fold.
    random_seed : int, default 0
        Base seed for reproducible column permutations.
    delta_cap : float or None, default None
        If set, clip each repeat's ΔQLIKE to ``[-delta_cap, delta_cap]``
        before averaging into the fold-level importance.

    Methods
    -------
    validate()
        Raise if options are invalid.
    describe()
        Return a short human-readable summary.
    """

    n_repeats: int = DEFAULT_N_REPEATS
    random_seed: int = DEFAULT_RANDOM_SEED
    delta_cap: float | None = None

    def validate(self) -> None:
        """Raise ``DataValidationError`` when options are invalid."""
        if self.n_repeats < 1:
            raise DataValidationError("n_repeats must be at least 1.")
        if self.delta_cap is not None and self.delta_cap <= 0:
            raise DataValidationError("delta_cap must be positive when set.")

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact description of importance options.
        """
        cap_text = "none" if self.delta_cap is None else f"{self.delta_cap}"
        return (
            f"n_repeats={self.n_repeats}, random_seed={self.random_seed}, "
            f"delta_cap={cap_text}"
        )


@dataclass(frozen=True, slots=True)
class TestWindow:
    """Held-out test slice for one walk-forward fold.

    Parameters
    ----------
    x_test : pandas.DataFrame
        Test feature rows.
    y_test : pandas.Series
        Test target values.
    fold_id : int
        Fold identifier.

    Methods
    -------
    row_count()
        Return the number of test rows.
    describe()
        Return a short human-readable summary.
    """

    x_test: pd.DataFrame
    y_test: pd.Series
    fold_id: int

    def row_count(self) -> int:
        """Return the number of test rows.

        Returns
        -------
        int
            Length of the test index.
        """
        return int(len(self.x_test))

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact description of the test window.
        """
        return f"fold_id={self.fold_id}, rows={self.row_count()}"


@dataclass(frozen=True, slots=True)
class FittedFoldState:
    """Fitted model and baseline score for one importance fold.

    Parameters
    ----------
    model : Any
        Model fitted on the training window only.
    window : TestWindow
        Held-out test slice.
    baseline_qlike : float
        QLIKE on unshuffled test predictions.
    options : ImportanceOptions
        Shuffle repeat settings.

    Methods
    -------
    repeat_count()
        Return the configured number of shuffle repeats.
    describe()
        Return a short human-readable summary.
    """

    model: Any
    window: TestWindow
    baseline_qlike: float
    options: ImportanceOptions

    def repeat_count(self) -> int:
        """Return the configured number of shuffle repeats.

        Returns
        -------
        int
            Value of ``options.n_repeats``.
        """
        return int(self.options.n_repeats)

    def describe(self) -> str:
        """Return a short human-readable summary.

        Returns
        -------
        str
            Compact description of the fitted fold state.
        """
        return (
            f"{self.window.describe()}, "
            f"baseline_qlike={self.baseline_qlike:.6f}"
        )


def permutation_importance_folds(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: ModelFactory,
    fold_spec: WalkForwardSpec,
    options: ImportanceOptions | None = None,
) -> pd.DataFrame:
    """Compute fold-wise permutation importance under QLIKE.

    For each expanding fold the model is fit on train only. Each feature
    column is shuffled on the **test** frame (never on train). Importance is
    ``QLIKE_permuted - QLIKE_baseline`` averaged over repeats. Higher values
    mean destroying the column hurt the forecast more.

    Parameters
    ----------
    features : pandas.DataFrame
        Predictor matrix indexed by session date.
    target : pandas.Series
        Realized-volatility target aligned to ``features``.
    model_factory : callable
        Zero-arg factory returning a fresh unfitted model.
    fold_spec : WalkForwardSpec
        Expanding walk-forward settings.
    options : ImportanceOptions or None, default None
        Repeat count and RNG seed (defaults used when ``None``).

    Returns
    -------
    pandas.DataFrame
        Long-form table with columns:
        ``fold_id``, ``feature``, ``importance``, ``baseline_qlike``,
        ``n_repeats``.

    Raises
    ------
    DataValidationError
        If inputs/settings are invalid or no folds can be scored.
    """
    fold_spec.validate()
    resolved = options if options is not None else ImportanceOptions()
    resolved.validate()

    feature_frame, target_series = _align_features_and_target(features, target)
    folds = generate_expanding_folds(
        index=feature_frame.index,
        n_splits=fold_spec.n_splits,
        embargo_size=fold_spec.embargo_size,
    )
    records = _importance_records_for_folds(
        feature_frame,
        target_series,
        model_factory,
        folds,
        resolved,
    )
    if not records:
        raise DataValidationError("No permutation-importance records were produced.")
    return pd.DataFrame.from_records(records)


def _importance_records_for_folds(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: ModelFactory,
    folds: list[WalkForwardFold],
    options: ImportanceOptions,
) -> list[dict[str, float | int | str]]:
    """Score permutation importance on every fold."""
    records: list[dict[str, float | int | str]] = []
    for fold in folds:
        records.extend(
            _importance_records_for_fold(
                features,
                target,
                model_factory,
                fold,
                options,
            )
        )
    return records


def _importance_records_for_fold(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: ModelFactory,
    fold: WalkForwardFold,
    options: ImportanceOptions,
) -> list[dict[str, float | int | str]]:
    """Fit once on train, then permute each test feature column."""
    state = _fit_fold_state(features, target, model_factory, fold, options)
    feature_names = list(features.columns)
    return [
        _importance_record(state, position, name)
        for position, name in enumerate(feature_names)
    ]


def _fit_fold_state(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: ModelFactory,
    fold: WalkForwardFold,
    options: ImportanceOptions,
) -> FittedFoldState:
    """Fit a fresh model on the training window and score the baseline."""
    window = TestWindow(
        x_test=features.loc[fold.test_index],
        y_test=target.loc[fold.test_index],
        fold_id=int(fold.fold_id),
    )
    model = model_factory()
    model.fit(features.loc[fold.train_index], target.loc[fold.train_index])
    baseline_qlike = qlike(window.y_test, model.predict(window.x_test))
    return FittedFoldState(
        model=model,
        window=window,
        baseline_qlike=float(baseline_qlike),
        options=options,
    )


def _importance_record(
    state: FittedFoldState,
    feature_position: int,
    feature_name: str,
) -> dict[str, float | int | str]:
    """Build one fold/feature importance record."""
    deltas = _qlike_deltas_for_feature(state, feature_name, feature_position)
    return {
        "fold_id": int(state.window.fold_id),
        "feature": feature_name,
        "importance": float(np.mean(deltas)),
        "baseline_qlike": float(state.baseline_qlike),
        "n_repeats": state.repeat_count(),
    }


def _qlike_deltas_for_feature(
    state: FittedFoldState,
    feature_name: str,
    feature_position: int,
) -> list[float]:
    """Return per-repeat QLIKE deltas for one shuffled feature."""
    deltas: list[float] = []
    for repeat_index in range(state.repeat_count()):
        shuffled = _shuffle_test_column(
            state,
            feature_name,
            feature_position,
            repeat_index,
        )
        permuted_qlike = qlike(state.window.y_test, state.model.predict(shuffled))
        delta = float(permuted_qlike - state.baseline_qlike)
        deltas.append(_maybe_cap_delta(delta, state.options.delta_cap))
    return deltas


def _maybe_cap_delta(delta: float, delta_cap: float | None) -> float:
    """Clip a QLIKE delta when a positive cap is configured."""
    if delta_cap is None:
        return delta
    return float(np.clip(delta, -delta_cap, delta_cap))


def _shuffle_test_column(
    state: FittedFoldState,
    feature_name: str,
    feature_position: int,
    repeat_index: int,
) -> pd.DataFrame:
    """Copy the test frame and shuffle one feature column."""
    seed = (
        state.options.random_seed
        + FOLD_SEED_STRIDE * state.window.fold_id
        + FEATURE_SEED_STRIDE * feature_position
        + repeat_index
    )
    rng = np.random.default_rng(seed)
    shuffled = state.window.x_test.copy()
    values = shuffled[feature_name].to_numpy(copy=True)
    rng.shuffle(values)
    shuffled[feature_name] = values
    return shuffled
