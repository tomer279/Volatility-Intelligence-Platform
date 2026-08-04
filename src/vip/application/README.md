# `vip.application`

## Purpose
The application package contains use-cases that orchestrate domain contracts and infrastructure adapters.  
CLI commands should call these functions rather than embedding business logic.

## Modules
- `ingest_market_data.py` - Fetch, validate, and persist daily OHLCV market data.
- `build_feature_matrix.py` - Load OHLCV, build features/target, and persist the matrix.
- `run_baseline_experiment.py` - Walk-forward baseline evaluation and artifact persistence.
- `screen_factors.py` - Factor screen (horse-race, OOS inference vs HAR, importance, HTML report).
- `screen_batch.py` - Multi-symbol batch screening with ingest/features caching.
- `screen_multi_horizon.py` - Multi-horizon factor screen orchestration + cross-horizon summary.
- `run_study.py` - Composite ingest → features → screen for one or more symbols.
- `__init__.py` - Public application exports.


## Key APIs
- `ingest_market_data(source, store, symbol, date_range)` - Run the ingestion pipeline.
- `IngestMarketDataResult` - Summary of a completed ingestion run.
- `build_and_persist_feature_matrix(market_store, feature_store, symbol, ...)` - Build and save features.
- `BuildFeatureMatrixResult` - Summary of a completed feature-matrix build.
- `run_baseline_experiment(feature_store, artifact_store, symbol, ...)` - Evaluate baselines and write metrics.
- `BaselineExperimentResult` - Summary table, fold metrics, and winning model.
- `screen_factors(feature_store, artifact_store, symbol, config=None, inference=None)` - Factor screen + inference + artifacts.
- `FactorScreenResult` / `ScreenConfig` / `ScreenInferenceOptions` - Result object and nested settings.
- `target_column_for_horizon(horizon_days)` - Returns ``target_rv_cc_{h}d``.
- `settings_for_horizon(horizon_days)` - M8 defaults: ``embargo_size = h``, horizon-aware bootstrap block length/bounds, ``horizon_days`` for NW.
- `ScreenArtifactContext` - Persist-time screen + inference settings for artifacts / `screen_meta`.
- `FeatureMatrixExtras` - Optional settings (`feature_names`, `include_vix`) for feature builds.
- `run_screen_batch(source, market_store, feature_store, artifact_store, config)` - Loop symbols: ingest/features/screen.
- `BatchScreenConfig` / `BatchScreenResult` - Batch settings and summary table.
- `run_study(stores, config)` - Full study pipeline; returns `BatchScreenResult`.
- `RunStudyConfig` - Symbols, date range, horizon, VIX flag, skip flags.
- `RunStudyStores` - Bundled source + market/feature/artifact stores.
- `screen_multi_horizon(stores, config)` - Run per-horizon `screen_factors` and write study artifacts.
- `MultiHorizonStores` / `MultiHorizonScreenConfig` / `MultiHorizonInferenceOverrides` /
  `MultiHorizonScreenResult` - Study dependencies, settings, inference overrides, and result.

## Dependencies
- Depends on: domain value objects/protocols, persistence stores, features pipeline, modeling baselines, evaluation walk-forward / inference, ingestion adapters (via injected source), reporting.
- Must not depend on: Typer/CLI formatting details.

## Usage
Ingestion:
1. CLI builds `YFinanceMarketDataSource` + `ParquetMarketDataStore`.
2. CLI calls `ingest_market_data(...)`.
3. Use-case returns `IngestMarketDataResult`.

Features:
1. CLI builds raw `ParquetMarketDataStore` + processed `ParquetFeatureMatrixStore`.
2. CLI calls `build_and_persist_feature_matrix(...)`.
3. Use-case returns `BuildFeatureMatrixResult`.

Baselines:
1. CLI builds `ParquetFeatureMatrixStore` + `FilesystemArtifactStore`.
2. CLI calls `run_baseline_experiment(...)`.
3. Use-case returns `BaselineExperimentResult` and writes `metrics.json` / `folds.json`.

Factor screen:
1. CLI builds feature + artifact stores.
2. CLI or orchestrator calls `screen_factors(...)`.
   - Default (omit `inference`): single-horizon h=5, target `target_rv_cc_5d`, embargo=5.
   - Non-default horizon: `defaults = settings_for_horizon(h)`, then
     `screen_factors(..., config=defaults.config, inference=defaults.inference)`
     (or `dataclasses.replace` for cheaper test settings). Target column is always
     derived from `inference.horizon_days` as `target_rv_cc_{h}d`.
3. Writes `metrics.json` (inference-enriched), `folds.json`, `oos_losses.json`,
   `inference.json`, optional `inference_sensitivity.json`, `importance.json`,
   `factor_ranking.json`, `metrics_by_regime.json`, `screen_meta.json`,
   `importance_plot.png`, and `report.html`.

Batch screen:
1. CLI builds stores + source + `BatchScreenConfig`.
2. CLI calls `run_screen_batch(...)`.
3. Per-symbol: ingest if missing, build features for `horizon_days` if missing,
   then screen with matching embargo / inference (via `settings_for_horizon`).
4. Returns `BatchScreenResult` with a summary DataFrame.

Full study (`vip run`):
1. CLI builds `RunStudyStores` (yfinance source + Parquet/artifact stores).
2. CLI builds `RunStudyConfig` from flags (`--symbol` / `--symbols`, `--with-vix`, skip flags).
3. CLI calls `run_study(stores, config)`.
4. Returns `BatchScreenResult` (summary table + per-symbol experiment IDs).

Multi-horizon screen (`vip screen-horizons`):
1. CLI builds `MultiHorizonStores` + `MultiHorizonScreenConfig`.
2. For each `h` in `{1,5,21}` (or `--horizons`): rebuild/load features for
   `target_rv_cc_{h}d`, call `settings_for_horizon(h)` + `screen_factors`,
   promote artifacts into `h{h}d/`.
3. Write study-level `screen_meta.json`, `horizon_summary.json`, and `report.html`
   under `data/artifacts/multi-horizon-screen-{symbol}-{date}/`.
4. Per-horizon artifacts are promoted under `h{h}d/` (metrics, OOS losses,
   inference, importance, optional per-horizon report pieces). Study-level
   `report.html` includes the **Skill by horizon** section.
5. Jump-robust features are **not** toggled by `MultiHorizonScreenConfig`
   today; builds use `FeatureMatrixExtras(include_vix=...)` only. To include
   jump columns, build matrices offline with
   `create_default_registry(include_jump=True)` (stretch; see methodology §2.6 / §11.5).

## Notes
- Keep use-cases framework-agnostic and easy to unit-test with fakes.
- Prefer dependency injection of source/store over constructing them inside the use-case.
- Screen inference defaults: block bootstrap primary vs `har_rv_ols`; optional HLN–DM;
  optional non-overlapping horizon subsample footnote (`inference_sensitivity.json`).
- For multi-horizon callers (M8), prefer `settings_for_horizon(h)` over hand-wiring
  embargo / block length; do not hard-code `target_rv_cc_5d`. Study orchestration is
  `screen_multi_horizon` (`vip screen-horizons`); single-horizon `vip screen` stays default h=5.
- M8 reuses `screen_factors` once per horizon — do not invent a second
  walk-forward/inference stack.