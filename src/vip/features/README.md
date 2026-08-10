# `vip.features`

## Purpose
Build predictive features and realized-volatility targets from canonical daily OHLCV data, with strict temporal alignment (no leakage).

## Modules
- `targets.py` - Forward realized-volatility labels.
- `realized.py` - Trailing RV helpers plus daily bipower / jump-proportion proxies.
- `returns.py` - Return-based features.
- `har.py` - HAR-style trailing RV features.
- `jump_features.py` - Opt-in jump-robust daily feature family (M8 stretch).
- `iv_rv_features.py` - Opt-in IV−RV gap family (VIX daily vol + gaps; pipeline-composed).
- `range_features.py` - High/low range features.
- `volume_features.py` - Volume z-score features.
- `registry.py` - Named feature-family registry and default Milestone 2 set.
- `pipeline.py` - End-to-end feature-matrix builder.
- `cross_asset.py` - VIX and TNX (rates) level / 1d change via backward asof join.

## Key APIs
- `daily_log_returns(close)` - Daily log returns from close prices.
- `build_target_rv_cc(ohlcv, horizon_days=5)` - Forward close-to-close RV target.
- `realized_volatility_trailing(returns, window)` - Trailing RV ending at `t`.
- `bipower_volatility_trailing(returns, window)` - Daily bipower-vol proxy ending at `t`.
- `jump_proportion_trailing(returns, window)` - `max(0, RV - BPV) / RV` ending at `t`.
- `build_return_features(ohlcv)` - `ret_1d`, `ret_5d`.
- `build_har_features(ohlcv)` - `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d`.
- `build_jump_features(ohlcv)` - `jump_prop_*` at 1/5/21.
- `build_range_features(ohlcv)` - `range_1d`, `range_5d_mean`.
- `build_volume_features(ohlcv)` - `volume_z_21d`.
- `create_default_registry(include_jump=False)` - Core families; set `include_jump=True` for `jump`.
- `FeatureRegistry.build_all(ohlcv, names=None)` - Assemble selected feature families.
- `build_vix_features(primary_index, vix_ohlcv)` - `vix_level`, `vix_chg_1d` via backward asof join.
- `build_rates_features(primary_index, rates_ohlcv)` - `tnx_level`, `tnx_chg_1d` via backward asof join.
- `VixJoinOptions(vix_ohlcv=None, include_iv_rv=False, rates_ohlcv=None)` - optional VIX / IV−RV / rates joins.
- `vix_level_to_daily_vol(vix_level)` - Locked `(vix_level / 100) / sqrt(252)`.
- `build_iv_rv_features(har_frame, vix_level)` - `vix_vol_daily`, `vix_minus_rv_*`, `vix_rv_ratio_5d`.
 `build_feature_matrix(ohlcv, horizon_days=5, ..., vix_ohlcv=None)` - Features + target (+ optional VIX / IV−RV / rates), NaNs dropped.

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
| `jump_prop_1d` | jump | Jump proportion at 1d window |
| `jump_prop_5d` | jump | Jump proportion at 5d window |
| `jump_prop_21d` | jump | Jump proportion at 21d window |
| `range_1d` | range | `(high_t - low_t) / close_t` |
| `range_5d_mean` | range | Mean of `range_1d` over last 5 sessions ending at `t` |
| `volume_z_21d` | volume | `(volume_t - mean_21) / std_21` trailing window ending at `t` |
| `vix_level` | cross_asset | VIX close as-of session `t` (backward `merge_asof`) |
| `vix_chg_1d` | cross_asset | VIX close pct-change as-of `t` (computed on VIX calendar, then asof-joined) |
| `vix_vol_daily` | iv_rv | `(vix_level / 100) / sqrt(252)` after as-of align |
| `vix_minus_rv_1d` | iv_rv | `vix_vol_daily − rv_cc_1d` |
| `vix_minus_rv_5d` | iv_rv | `vix_vol_daily − rv_cc_5d` |
| `vix_minus_rv_21d` | iv_rv | `vix_vol_daily − rv_cc_21d` |
| `vix_rv_ratio_5d` | iv_rv | `vix_vol_daily / rv_cc_5d` (NaN if `rv_cc_5d` ≤ 0) |
| `tnx_level` | cross_asset (rates) | TNX close (yield %) as-of session `t` (backward `merge_asof`) |
| `tnx_chg_1d` | cross_asset (rates) | TNX close pct-change as-of `t` (on TNX calendar, then asof-joined) |


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
- Bipower variation is computed internally for `jump_prop_*` only; `bpv_cc_*` level columns are **not** exported (they duplicate HAR `rv_cc_*` and destabilize permutation importance).
- `--with jump` adds **3** columns (`jump_prop_1d/5d/21d`), not 6.
- IV−RV family is opt-in: `vip features --with iv_rv` (implies VIX load),
  `FeatureMatrixExtras(include_iv_rv=True)`, or
  `VixJoinOptions(vix_ohlcv=..., include_iv_rv=True)` in the pipeline.
- Bare `--with vix` adds only `vix_level` / `vix_chg_1d` (no gap columns).
- `iv_rv` is pipeline-composed after HAR + as-of VIX; not an OHLCV `FeatureSpec`
  in `create_default_registry`.
- `--with iv_rv` adds **5** columns (`vix_vol_daily`, three gaps, `vix_rv_ratio_5d`)
  plus the usual 2 VIX columns when VIX is joined.
- IV−RV gaps are a **research proxy** (aligned daily-vol scale), not
  variance-swap / options-replication identities; see
  `docs/research_methodology.md` §2.7 and §12.
- Gap columns feed Ridge/Lasso screening; the competing forecast model
  `vix_as_forecast` uses `vix_vol_daily` / `vix_level` only (not the gap vector).
- Rates family is opt-in: `vip features --with rates` after `vip ingest --symbol TNX`,
  or `FeatureMatrixExtras(include_rates=True)`. Storage symbol `TNX` maps to Yahoo `^TNX`.
- `--with rates` adds **2** columns (`tnx_level`, `tnx_chg_1d`). Yield is a **level/change
  covariate**, not converted with the VIX daily-vol formula.
- Rates may be combined with other tokens, e.g. `--with vix,iv_rv,rates`.