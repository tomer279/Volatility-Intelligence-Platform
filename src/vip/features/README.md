# `vip.features`

## Purpose
Build predictive features and realized-volatility targets from canonical daily OHLCV data, with strict temporal alignment (no leakage).

## Modules
- `targets.py` - Forward realized-volatility labels.
- `realized.py` - Trailing RV helpers plus daily bipower / jump-proportion proxies.
- `returns.py` - Return-based features.
- `har.py` - HAR-style trailing RV features.
- `jump_features.py` - Opt-in jump-robust daily feature family (M8 stretch).
- `range_features.py` - High/low range features.
- `volume_features.py` - Volume z-score features.
- `registry.py` - Named feature-family registry and default Milestone 2 set.
- `pipeline.py` - End-to-end feature-matrix builder.
- `cross_asset.py` - VIX level / 1d change joined onto the primary calendar.

## Key APIs
- `daily_log_returns(close)` - Daily log returns from close prices.
- `build_target_rv_cc(ohlcv, horizon_days=5)` - Forward close-to-close RV target.
- `realized_volatility_trailing(returns, window)` - Trailing RV ending at `t`.
- `bipower_volatility_trailing(returns, window)` - Daily bipower-vol proxy ending at `t`.
- `jump_proportion_trailing(returns, window)` - `max(0, RV - BPV) / RV` ending at `t`.
- `build_return_features(ohlcv)` - `ret_1d`, `ret_5d`.
- `build_har_features(ohlcv)` - `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d`.
- `build_jump_features(ohlcv)` - `bpv_cc_*` and `jump_prop_*` at 1/5/21.
- `build_range_features(ohlcv)` - `range_1d`, `range_5d_mean`.
- `build_volume_features(ohlcv)` - `volume_z_21d`.
- `create_default_registry(include_jump=False)` - Core families; set `include_jump=True` for `jump`.
- `FeatureRegistry.build_all(ohlcv, names=None)` - Assemble selected feature families.
- `build_vix_features(primary_index, vix_ohlcv)` - `vix_level`, `vix_chg_1d` via backward asof join.
- `build_feature_matrix(ohlcv, horizon_days=5, ..., vix_ohlcv=None)` - Features + target (+ optional VIX), NaNs dropped.

## Research contract
- Features at session `t` use information with timestamp `<= t`.
- Target at session `t` uses returns over `t+1 .. t+h` only.
- Primary defaults: `h = 5`, close-to-close RV, column `target_rv_cc_5d`.
- Target stored **non-annualized**.
- Incomplete rows (NaN features/target) are dropped by `build_feature_matrix`.

## Column dictionary

| Column | Family | Definition |
|--------|--------|------------|
| `target_rv_cc_5d` | target | `sqrt(sum_{i=1..5} r_{t+i}^2)` where `r` is daily log return |
| `ret_1d` | returns | `log(close_t / close_{t-1})` |
| `ret_5d` | returns | `log(close_t / close_{t-5})` |
| `rv_cc_1d` | har | Trailing 1d RV: `sqrt(r_t^2)` |
| `rv_cc_5d` | har | Trailing 5d RV ending at `t` |
| `rv_cc_21d` | har | Trailing 21d RV ending at `t` |
| `bpv_cc_1d` | jump | Trailing 1d daily bipower vol (not tick bipower) |
| `bpv_cc_5d` | jump | Trailing 5d daily bipower vol |
| `bpv_cc_21d` | jump | Trailing 21d daily bipower vol |
| `jump_prop_1d` | jump | Jump proportion at 1d window |
| `jump_prop_5d` | jump | Jump proportion at 5d window |
| `jump_prop_21d` | jump | Jump proportion at 21d window |
| `range_1d` | range | `(high_t - low_t) / close_t` |
| `range_5d_mean` | range | Mean of `range_1d` over last 5 sessions ending at `t` |
| `volume_z_21d` | volume | `(volume_t - mean_21) / std_21` trailing window ending at `t` |
| `vix_level` | cross_asset | VIX close as-of session `t` (backward `merge_asof`) |
| `vix_chg_1d` | cross_asset | VIX close pct-change as-of `t` (computed on VIX calendar, then asof-joined) |

## Notes
- Source OHLCV should be valid daily bars; `build_feature_matrix` re-validates via ingestion validators.
- HAR columns are **trailing** features; do not confuse them with the forward target `target_rv_cc_5d`.
- Output path for persisted matrices (Milestone 2 later steps): `data/processed/{SYMBOL}/features.parquet`.
- VIX is optional: pass `vix_ohlcv` into the pipeline, or use `vip features --with vix` after `vip ingest --symbol VIX`.
- Cross-asset joins never use future VIX prints (`direction="backward"` only).
- Jump family is opt-in: `vip features --with jump` (or `--with vix,jump`),
  `FeatureMatrixExtras(include_jump=True)`, or
  `create_default_registry(include_jump=True)` then `build_all` / `build_feature_matrix`.
- Jump estimators are **daily close-to-close proxies**, not high-frequency / tick bipower; see `docs/research_methodology.md` §2.6.