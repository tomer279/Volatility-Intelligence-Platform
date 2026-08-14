# `vip.application`

## Purpose
The application package contains use-cases that orchestrate domain contracts and infrastructure adapters.  
CLI commands should call these functions rather than embedding business logic.

## Modules
- `ingest_market_data.py` - Fetch, validate, and persist daily OHLCV market data.
- `build_feature_matrix.py` - Load OHLCV, build features/target, and persist the matrix.
- `run_baseline_experiment.py` - Walk-forward baseline evaluation and artifact persistence.
- `screen_factors.py` - Factor-screen orchestration (load matrix, importance, regimes, assemble result).
- `screen_horse_race.py` - Horse-race model catalog, VIX-conditional resolve, walk-forward + M7 inference.
- `screen_factor_artifacts.py` - Persist screen JSON / plots / HTML report for one experiment.
- `screen_batch.py` - Multi-symbol batch screening with ingest caching; features rebuild unless skip.
- `screen_multi_horizon.py` - Multi-horizon factor screen orchestration + cross-horizon summary.
- `run_study.py` - Composite ingest → features → screen for one or more symbols.
- `__init__.py` - Public application exports.

## Key APIs
- `ingest_market_data(source, store, symbol, date_range)` - Run the ingestion pipeline.
- `IngestMarketDataResult` - Summary of a completed ingestion run.
- `build_and_persist_feature_matrix(market_store, feature_store, symbol, ...)` - Build and save features.
- `BuildFeatureMatrixResult` - Summary of a completed feature-matrix build.
- `FeatureMatrixExtras` - Optional settings (`feature_names`, `include_vix`,
  `include_jump`, `include_iv_rv`, `vix_symbol`) for feature builds.
  `include_iv_rv=True` implies VIX load.
- `run_baseline_experiment(feature_store, artifact_store, symbol, ...)` - Evaluate baselines and write metrics.
- `BaselineExperimentResult` - Summary table, fold metrics, and winning model.
- `screen_factors(feature_store, artifact_store, symbol, config=None, inference=None)` - Factor screen + inference + artifacts.
- `FactorScreenResult` / `ScreenConfig` / `ScreenInferenceOptions` - Result object and nested settings.
- `target_column_for_horizon(horizon_days)` - Returns ``target_rv_cc_{h}d``.
- `settings_for_horizon(horizon_days)` - M8 defaults: ``embargo_size = h``, horizon-aware bootstrap block length/bounds, ``horizon_days`` for NW.
- `ScreenArtifactContext` - Persist-time screen + inference settings for artifacts / `screen_meta`.
- `HORSE_RACE_MODELS` / `VIX_AS_FORECAST_MODEL` / `OU_RV_MODEL` /
  `EWMA_RECURSIVE_MODEL` / `HorseRaceOptions` /
  `run_horse_race_with_inference` / `resolve_horse_race_models` - Horse-race catalog and runner
  (`screen_horse_race.py`); ``vix_as_forecast`` is omitted when the matrix lacks
  ``vix_vol_daily`` / ``vix_level``. ``ou_rv`` and ``ewma_recursive`` are
  always in the race; ``ou_rv`` ``horizon_days`` comes from
  ``ScreenInferenceOptions`` (via ``summary_options.horizon_days``).
- `persist_screen_artifacts(artifact_store, result, context)` - Write screen JSON, plots, and `report.html`
  (`screen_factor_artifacts.py`); called by `screen_factors`.
- `run_screen_batch(source, market_store, feature_store, artifact_store, config)` - Loop symbols: ingest/features/screen.
- `BatchScreenConfig` / `BatchScreenResult` - Batch settings and summary table.
- `run_study(stores, config)` - Full study pipeline; returns `BatchScreenResult`.
- `RunStudyConfig` - Symbols, date range, horizon, `extras` (`FeatureMatrixExtras`), skip flags.
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
3. Internally: `run_horse_race_with_inference` (models from `HORSE_RACE_MODELS`) →
   importance / regimes → `persist_screen_artifacts`.
4. Writes `metrics.json` (inference-enriched), `folds.json`, `oos_losses.json`,
   `inference.json`, optional `inference_sensitivity.json`, `importance.json`,
   `factor_ranking.json`, `metrics_by_regime.json`, `screen_meta.json`,
   `importance_plot.png`, and `report.html`.

Batch screen:
1. CLI builds stores + source + `BatchScreenConfig`.
2. CLI calls `run_screen_batch(...)`.
3. Per-symbol: ingest if missing; **always rebuild** features for
   `horizon_days` unless `--skip-features` (then require
   `target_rv_cc_{h}d`); screen via `settings_for_horizon`.
4. Returns `BatchScreenResult` with a summary DataFrame.

Full study (`vip run`):
1. CLI builds `RunStudyStores` (yfinance source + Parquet/artifact stores).
2. CLI builds `RunStudyConfig` from flags (`--symbol` / `--symbols`, `--with`, skip flags).
3. CLI calls `run_study(stores, config)`.
4. Returns `BatchScreenResult` (summary table + per-symbol experiment IDs).

Feature step always rebuilds unless `--skip-features` so horizon/extras
from the CLI are not silently ignored when a stale parquet exists.

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
5. VIX / jump / IV−RV / rates extras: CLI `--with vix,jump,iv_rv,rates` → `parse_feature_extras` →
   `MultiHorizonScreenConfig(feature_extras=...)` →
   `build_and_persist_feature_matrix(..., extras=config.feature_extras)`.
   No effect under `--skip-features` unless those columns already exist in the matrix.
   Note: all horizons share one `features.parquet` path; the last horizon
   processed wins. Subsequent `vip run` / `screen-batch` without
   `--skip-features` rebuild for their own horizon.

## Notes
- Keep use-cases framework-agnostic and easy to unit-test with fakes.
- Prefer dependency injection of source/store over constructing them inside the use-case.
- Screen inference defaults: block bootstrap primary vs `har_rv_ols`; optional HLN–DM;
  optional non-overlapping horizon subsample footnote (`inference_sensitivity.json`).
- Horse-race catalog includes `ou_rv` and `ewma_recursive` unconditionally
  and `vix_as_forecast` only when the feature matrix has `vix_vol_daily`
  or `vix_level` (typical with `--with vix` / `iv_rv`).
- IV−RV gap columns appear only when the **persisted** matrix was built with
  `include_iv_rv=True`. `screen_factors` does not rebuild extras; rebuild via
  `build_and_persist_feature_matrix` / CLI `features` / `run` first.
- For multi-horizon callers (M8), prefer `settings_for_horizon(h)` over hand-wiring
  embargo / block length; do not hard-code `target_rv_cc_5d`. Study orchestration is
  `screen_multi_horizon` (`vip screen-horizons`); single-horizon `vip screen` stays default h=5.
- M8/M9 reuse `screen_factors` (and its horse-race / artifact helpers) — do not invent a
  second walk-forward/inference stack.
- Rates columns appear only when the persisted matrix was built with
  `include_rates=True` (same rebuild rule as IV−RV).