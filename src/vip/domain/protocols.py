"""Structural interfaces (protocols) for VIP adapters.

Exports
-------
MarketDataSource
    Fetches market bars for a symbol and date range.
MarketDataStore
    Persists and loads normalized OHLCV tables.
FeatureBuilder
    Builds a feature matrix from market data.
VolatilityModel
    Fits and predicts realized-volatility targets.
Metric
    Scores predictions against realized outcomes.
ArtifactStore
    Writes and reads experiment artifacts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from vip.domain.value_objects import DateRange, ExperimentId, Symbol


@runtime_checkable
class MarketDataSource(Protocol):
    """Fetches market data from an external vendor.

    Methods
    -------
    fetch(symbol, date_range)
        Download or read bars for the request window.
    source_name()
        Return a stable vendor identifier.
    """

    def fetch(self, symbol: Symbol, date_range: DateRange) -> pd.DataFrame:
        """Fetch OHLCV (or equivalent) bars.

        Parameters
        ----------
        symbol : Symbol
            Instrument to fetch.
        date_range : DateRange
            Inclusive calendar request window.

        Returns
        -------
        pandas.DataFrame
            Normalized market bars indexed by session date.
        """

    def source_name(self) -> str:
        """Return a stable vendor identifier.

        Returns
        -------
        str
            For example ``'yfinance'``.
        """


@runtime_checkable
class MarketDataStore(Protocol):
    """Persists normalized market data locally.

    Methods
    -------
    save(symbol, frame)
        Write a market table for ``symbol``.
    load(symbol)
        Read a previously saved market table.
    exists(symbol)
        Return whether data for ``symbol`` is present.
    """

    def save(self, symbol: Symbol, frame: pd.DataFrame) -> None:
        """Persist a market data table.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.
        frame : pandas.DataFrame
            Normalized OHLCV table.
        """

    def load(self, symbol: Symbol) -> pd.DataFrame:
        """Load a persisted market data table.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.

        Returns
        -------
        pandas.DataFrame
            Normalized OHLCV table.
        """

    def exists(self, symbol: Symbol) -> bool:
        """Return whether data for a symbol exists on disk.

        Parameters
        ----------
        symbol : Symbol
            Instrument key.

        Returns
        -------
        bool
            True when a dataset is available.
        """


@runtime_checkable
class FeatureBuilder(Protocol):
    """Transforms market data into model features.

    Methods
    -------
    build(market_data)
        Construct the feature matrix.
    name()
        Return the builder registry name.
    """

    def build(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Build a feature matrix from market data.

        Parameters
        ----------
        market_data : pandas.DataFrame
            Normalized market bars.

        Returns
        -------
        pandas.DataFrame
            Feature columns aligned to the prediction time index.
        """

    def name(self) -> str:
        """Return the builder registry name.

        Returns
        -------
        str
            Stable feature-builder identifier.
        """


@runtime_checkable
class VolatilityModel(Protocol):
    """Supervised model for realized-volatility forecasting.

    Methods
    -------
    fit(features, target)
        Fit on a training window.
    predict(features)
        Predict realized volatility for rows in ``features``.
    """

    def fit(self, features: pd.DataFrame, target: pd.Series) -> VolatilityModel:
        """Fit the model on training data.

        Parameters
        ----------
        features : pandas.DataFrame
            Training features.
        target : pandas.Series
            Training realized-volatility target.

        Returns
        -------
        VolatilityModel
            Fitted model (typically ``self``).
        """

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict realized volatility.

        Parameters
        ----------
        features : pandas.DataFrame
            Feature rows to score.

        Returns
        -------
        pandas.Series
            Predicted volatility values.
        """


@runtime_checkable
class Metric(Protocol):
    """Scalar score for volatility forecasts.

    Methods
    -------
    name()
        Return the metric identifier.
    compute(y_true, y_pred)
        Compute the metric value.
    """

    def name(self) -> str:
        """Return the metric identifier.

        Returns
        -------
        str
            For example ``'qlike'``.
        """

    def compute(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Compute the metric value.

        Parameters
        ----------
        y_true : pandas.Series
            Realized target values.
        y_pred : pandas.Series
            Predicted values.

        Returns
        -------
        float
            Metric score (interpretation depends on the metric).
        """


@runtime_checkable
class ArtifactStore(Protocol):
    """Stores experiment outputs such as metrics and reports.

    Methods
    -------
    write_json(experiment_id, name, payload)
        Persist a JSON-serializable artifact.
    read_json(experiment_id, name)
        Load a previously written JSON artifact.
    """

    def write_json(
        self,
        experiment_id: ExperimentId,
        name: str,
        payload: dict[str, object],
    ) -> None:
        """Persist a JSON-serializable artifact.

        Parameters
        ----------
        experiment_id : ExperimentId
            Experiment namespace.
        name : str
            Artifact basename without extension.
        payload : dict of str to object
            JSON-serializable mapping.
        """

    def read_json(
        self,
        experiment_id: ExperimentId,
        name: str,
    ) -> dict[str, object]:
        """Load a JSON artifact.

        Parameters
        ----------
        experiment_id : ExperimentId
            Experiment namespace.
        name : str
            Artifact basename without extension.

        Returns
        -------
        dict of str to object
            Loaded payload.
        """
