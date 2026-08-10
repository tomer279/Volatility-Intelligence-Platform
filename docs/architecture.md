# Architecture

This document describes the Volatility Intelligence Platform's internal
architecture for developers joining the project.

---

## 1  Layer Diagram

```
┌───────────────────────────────────────────────┐
│                     cli                       │  ← user entry point (Typer)
├───────────────────────────────────────────────┤
│                  application                  │  ← use-case orchestration
├───────────────────────────────────────────────┤
│                  reporting                    │  ← HTML memo generation
├───────────────────────────────────────────────┤
│                visualization                  │  ← matplotlib research plots
├───────────────────────────────────────────────┤
│                  evaluation                   │  ← metrics, walk-forward, importance
├───────────────────────────────────────────────┤
│                   modeling                    │  ← baselines, linear, tree models
├───────────────────────────────────────────────┤
│                   features                    │  ← target & feature pipeline
├───────────────────────────────────────────────┤
│                  ingestion                    │  ← yfinance adapter, validation
├───────────────────────────────────────────────┤
│                 persistence                   │  ← Parquet stores, artifact I/O
├───────────────────────────────────────────────┤
│                    config                     │  ← YAML loading, AppConfig
├───────────────────────────────────────────────┤
│                    domain                     │  ← protocols, entities, errors
└───────────────────────────────────────────────┘
```

**Dependency direction:** layers depend only downward.  `domain` has zero
internal imports.  `cli` may import anything; `domain` imports nothing from
`vip`.

`orchestration` (logging utilities) is a cross-cutting concern used at all
layers.

---

## 2  Key Protocols

Protocols live in `vip.domain.protocols` and are `@runtime_checkable`.

### MarketDataSource

```python
class MarketDataSource(Protocol):
    def fetch(self, symbol: Symbol, date_range: DateRange) -> pd.DataFrame: ...
    def source_name(self) -> str: ...
```

**Implementation:** `YFinanceMarketDataSource` (in `vip.ingestion`).

### VolatilityModel

```python
class VolatilityModel(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "VolatilityModel": ...
    def predict(self, features: pd.DataFrame) -> pd.Series: ...
```

**Implementations:** `HistoricalMeanModel`, `EwmaModel`, `HarRvOlsModel`,
`RidgeModel`, `LassoModel`, `ElasticNetModel`, `RandomForestVolModel`
(in `vip.modeling`).

Other protocols: `MarketDataStore`, `FeatureBuilder`, `Metric`,
`ArtifactStore` — each with one or two concrete implementations.

---

## 3  Data Flow

```
Raw OHLCV (yfinance)
  │
  ▼
validate_and_normalize_ohlcv()          ← canonical schema
  │
  ▼
ParquetMarketDataStore.save()           ← data/market/{SYMBOL}.parquet
  │
  ▼
FeatureRegistry.build_all()             ← feature columns (returns, HAR, range, volume)
  + optional build_vix_features()       ← cross-asset VIX columns
  + build_target_rv_cc()                ← target_rv_cc_5d
  │
  ▼
ParquetFeatureMatrixStore.save()        ← data/features/{SYMBOL}.parquet
  │
  ▼
generate_expanding_folds()              ← 5 walk-forward folds with embargo
  │
  ▼
run_walk_forward() per model            ← OOS predictions + per-fold metrics
  │
  ▼
permutation_importance_folds()          ← ΔQLIKE per feature per fold
  + optional TreeSHAP                   ← mean |SHAP| per feature per fold
  │
  ▼
summarize_importance()                  ← median aggregation, rankings
  │
  ▼
FilesystemArtifactStore                 ← metrics.json, importance.json
  + plot_importance_bars()              ← importance_plot.png
  + render_factor_screen_report()       ← report.html
```

---

## 4  Package-by-Package Summary

**domain** — Value objects (`Symbol`, `DateRange`, `Horizon`, `ExperimentId`),
enums (`RvEstimator`, `MetricName`), typed errors (`VipError` hierarchy), and
the six `Protocol` definitions that all other layers code against.

**config** — Loads `configs/default.yaml` into a validated `AppConfig` dataclass.
Resolves the project root and provides path helpers.

**persistence** — Three concrete stores: `ParquetMarketDataStore` (OHLCV),
`ParquetFeatureMatrixStore` (feature matrices), and `FilesystemArtifactStore`
(JSON + binary experiment artifacts).  All are file-system-backed.

**orchestration** — `configure_logging` and `get_logger` utilities.  Provides
structured logging across all layers.

**ingestion** — `YFinanceMarketDataSource` fetches OHLCV from Yahoo Finance.
`validate_and_normalize_ohlcv` enforces the canonical schema (lowercase columns,
DatetimeIndex, no nulls, `high ≥ max(open, close, low)`, `volume ≥ 0`).

**features** — `FeatureRegistry` maps string names to `FeatureSpec` builder
callables.  `build_feature_matrix` orchestrates: validate → build features →
optionally add VIX columns → build target → concatenate → drop NaN.

**modeling** — Baseline models (`HistoricalMean`, `EWMA`, `HarRvOls`,
`VixAsForecast`), regularised linear models (`Ridge`, `Lasso`, `ElasticNet`
with per-fold `StandardScaler`), and `RandomForestVolModel`.  `ModelRegistry`
maps names to `ModelSpec` objects.

**evaluation** — Metrics (`qlike`, `mse`, `mae`), expanding walk-forward fold
generation with embargo, `run_walk_forward` execution, horse-race comparison via
`summarize_walk_forward`, permutation importance, and
`summarize_importance` (median aggregation, top-k hit-rate).

**visualization** — `plot_importance_bars` renders a horizontal bar chart of
ranked feature importance.  `apply_research_style` / `reset_research_style`
manage a consistent matplotlib theme.

**reporting** — Jinja2-based HTML report generation.  `build_factor_screen_context`
assembles the template context (methodology, horse-race table, ranked factors,
regime table, plot, caveats, Implied vs realized); `render_factor_screen_report`
produces the HTML string; `write_html_report` saves it to disk.

**application** — Thin use-case functions that compose lower layers:
`ingest_market_data`, `build_and_persist_feature_matrix`, `run_baseline_experiment`,
`screen_factors` (horse-race via `screen_horse_race`, persistence via
`screen_factor_artifacts`).  Each returns a typed result dataclass.

**cli** — Typer commands: `info`, `ingest`, `features`, `evaluate`, `screen`,
`screen-batch`, `run`.  Each command parses arguments, calls an application
use-case, and prints results.

---

## 5  Extension Points

### Adding a model

Register a new `ModelSpec` in `vip.modeling.registry`.  The model must satisfy
the `VolatilityModel` protocol (`fit` / `predict`).  It will automatically
participate in the walk-forward horse-race and permutation importance.

### Adding a feature

See [`how_to_add_feature.md`](how_to_add_feature.md) for a step-by-step guide.
In brief: implement a builder function, register a `FeatureSpec` in the
`FeatureRegistry`, write a leakage test, and rebuild the matrix.

### Adding a metric

Implement the `Metric` protocol (`name` / `compute`) and wire it into
`run_walk_forward`.
