"""Core domain entities for research datasets and experiments.

Exports
-------
Instrument
    Tradable symbol metadata.
DatasetRef
    Pointer to a persisted market dataset.
ExperimentSpec
    Minimal specification for a volatility forecast experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from vip.domain.enums import MetricName, RvEstimator
from vip.domain.value_objects import Horizon, Symbol


@dataclass(frozen=True, slots=True)
class Instrument:
    """Tradable equity or ETF used in a study.

    Parameters
    ----------
    symbol : Symbol
        Normalized ticker.
    currency : str, default ``'USD'``
        Quote currency code.

    Methods
    -------
    display_name()
        Return a short human-readable label.
    is_usd_quoted()
        Return whether the instrument is quoted in USD.
    """

    symbol: Symbol
    currency: str = "USD"

    def display_name(self) -> str:
        """Return a short human-readable label.

        Returns
        -------
        str
            Ticker string, optionally with currency when not USD.
        """
        if self.currency == "USD":
            return self.symbol.value
        return f"{self.symbol.value} ({self.currency})"

    def is_usd_quoted(self) -> bool:
        """Return whether the instrument is quoted in USD.

        Returns
        -------
        bool
            True when ``currency`` is ``'USD'``.
        """
        return self.currency == "USD"


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Reference to a persisted market dataset on disk.

    Parameters
    ----------
    symbol : Symbol
        Instrument the dataset belongs to.
    relative_path : str
        Path relative to the project data root
        (for example ``'raw/SPY/ohlcv.parquet'``).
    content_hash : str or None, default None
        Optional hash of file contents for reproducibility checks.

    Methods
    -------
    has_content_hash()
        Return whether a content hash is present.
    matches_symbol(raw_symbol)
        Return whether this dataset is for ``raw_symbol``.
    """

    symbol: Symbol
    relative_path: str
    content_hash: str | None = None

    def has_content_hash(self) -> bool:
        """Return whether a content hash is present.

        Returns
        -------
        bool
            True when ``content_hash`` is not None.
        """
        return self.content_hash is not None

    def matches_symbol(self, raw_symbol: str) -> bool:
        """Return whether this dataset belongs to a raw ticker.

        Parameters
        ----------
        raw_symbol : str
            Ticker string to compare.

        Returns
        -------
        bool
            True when ``symbol`` matches ``raw_symbol``.
        """
        return self.symbol.matches(raw_symbol)


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Minimal specification for a realized-volatility experiment.

    Full experiment configs live in YAML / ``AppConfig``. This entity
    captures the research identity used by domain and application code.

    Parameters
    ----------
    symbol : Symbol
        Target instrument.
    horizon : Horizon
        Forecast horizon in trading days.
    rv_estimator : RvEstimator
        Realized-volatility estimator used for the target.
    primary_metric : MetricName
        Metric used for model selection and ranking.

    Methods
    -------
    target_label()
        Return a compact target name for columns and reports.
    describes(symbol)
        Return whether this spec targets ``symbol``.
    """

    symbol: Symbol
    horizon: Horizon
    rv_estimator: RvEstimator
    primary_metric: MetricName

    def target_label(self) -> str:
        """Return a compact target name for columns and reports.

        Returns
        -------
        str
            Label such as ``'rv_close_to_close_5d'``.
        """
        return f"rv_{self.rv_estimator.value}_{self.horizon.label()}"

    def describes(self, symbol: Symbol) -> bool:
        """Return whether this spec targets a given symbol.

        Parameters
        ----------
        symbol : Symbol
            Symbol to compare.

        Returns
        -------
        bool
            True when symbols are equal.
        """
        return self.symbol == symbol
