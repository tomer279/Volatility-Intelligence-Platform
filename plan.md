# Volatility Intelligence Platform — Architecture Design

## 1. Overall Architecture

Use a **layered clean architecture** with a thin orchestration shell (CLI/notebooks/API later). Domain logic never depends on Yahoo Finance, scikit-learn, or Plotly; those live behind interfaces.

```
┌─────────────────────────────────────────────────────────────┐
│  Interfaces (CLI · Notebooks · optional FastAPI later)      │
├─────────────────────────────────────────────────────────────┤
│  Application / Orchestration (pipelines, use-cases)         │
├─────────────────────────────────────────────────────────────┤
│  Domain (entities, protocols, research contracts)           │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Ingestion   │  Features    │  Modeling    │  Evaluation    │
├──────────────┴──────────────┴──────────────┴────────────────┤
│  Visualization · Reporting · Persistence · Config           │
└─────────────────────────────────────────────────────────────┘
```

**Core research question:** which features best predict **future realized volatility** (e.g. next-day / next-week RV) for a given symbol?

Treat that as a **supervised forecasting problem** with strict temporal integrity (no leakage), walk-forward evaluation, and factor attribution — not a single “best model” demo.

### Layer responsibilities


| Layer              | Responsibility                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| **Domain**         | Types for bars, RV targets, feature sets, experiment specs; protocols for data sources, feature builders, models, metrics |
| **Application**    | Use-cases: ingest → build features → train/evaluate → report; experiment runners                                          |
| **Infrastructure** | Yahoo/Polygon clients, Parquet store, sklearn/LightGBM adapters, Plotly writers                                           |
| **Interfaces**     | CLI entrypoints, notebook helpers, config loading                                                                         |


**Dependency rule:** Interfaces → Application → Domain ← Infrastructure adapters. Infrastructure implements Domain protocols.

---

## 2. Suggested Directory Structure

```text
volatility-intelligence-platform/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── experiments/
│   │   ├── baseline_har.yaml
│   │   └── factor_screen_spy.yaml
│   └── symbols/
│       └── liquid_etfs.yaml
├── data/                         # gitignored; local cache
│   ├── raw/
│   ├── processed/
│   └── artifacts/                # models, reports, figures
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_research.ipynb
│   └── 03_model_diagnostics.ipynb
├── docs/
│   ├── architecture.md
│   ├── research_methodology.md
│   └── contributing.md
├── src/
│   └── vip/                      # package root (Volatility Intelligence Platform)
│       ├── __init__.py
│       ├── py.typed
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── entities.py       # Bar, Symbol, RealizedVolTarget, ...
│       │   ├── enums.py          # Frequency, Horizon, SplitMode
│       │   ├── protocols.py      # MarketDataSource, FeatureBuilder, ...
│       │   ├── errors.py
│       │   └── value_objects.py  # DateRange, ExperimentId
│       ├── application/
│       │   ├── __init__.py
│       │   ├── ingest_market_data.py
│       │   ├── build_feature_matrix.py
│       │   ├── run_experiment.py
│       │   ├── screen_factors.py
│       │   └── generate_report.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── yfinance_source.py
│       │   ├── polygon_source.py   # optional later
│       │   ├── validators.py
│       │   └── calendar.py         # trading calendar helpers
│       ├── features/
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── targets.py          # RV estimators & horizons
│       │   ├── returns.py
│       │   ├── realized.py         # RV, bipower, jump proxies
│       │   ├── har.py              # HAR-style lags
│       │   ├── technical.py
│       │   ├── calendar_features.py
│       │   ├── cross_asset.py      # VIX, SPY as optional covariates
│       │   └── pipeline.py
│       ├── modeling/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── baselines.py        # HAR OLS, RW, EWMA
│       │   ├── sklearn_models.py
│       │   ├── gradient_boosting.py
│       │   ├── regularization.py   # Ridge/Lasso/ElasticNet
│       │   ├── selection.py        # walk-forward feature selection
│       │   └── trainer.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py          # MSE, QLIKE, MAE, R², hit-rate
│       │   ├── splitting.py        # purged/embargoed time splits
│       │   ├── walk_forward.py
│       │   ├── importance.py       # permutation, SHAP adapter
│       │   ├── stability.py        # factor stability over time
│       │   └── comparison.py       # model/factor horse-race tables
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── styles.py
│       │   ├── timeseries.py
│       │   ├── residuals.py
│       │   ├── importance_plots.py
│       │   └── report_figures.py
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── templates/
│       │   ├── html_report.py
│       │   ├── markdown_report.py
│       │   └── experiment_summary.py
│       ├── persistence/
│       │   ├── __init__.py
│       │   ├── parquet_store.py
│       │   ├── artifact_store.py
│       │   └── metadata.py         # experiment lineage
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py           # pydantic settings
│       │   ├── loader.py
│       │   └── defaults.py
│       ├── orchestration/
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   ├── container.py        # DI wiring
│       │   └── logging.py
│       └── cli/
│           ├── __init__.py
│           ├── main.py
│           └── commands/
│               ├── ingest.py
│               ├── features.py
│               ├── train.py
│               ├── evaluate.py
│               └── report.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── contract/                 # protocol compliance tests
├── scripts/
│   ├── bootstrap_env.py
│   └── run_smoke_experiment.py
└── .github/
    └── workflows/
        └── ci.yml
```

Package name `vip` keeps imports short (`from vip.features import ...`) while the product name stays “Volatility Intelligence Platform”.

---

## 3. Modules and Responsibilities

### Domain (`vip.domain`)

- **entities:** immutable market bar series, feature matrices, prediction frames, experiment results  
- **protocols:** `MarketDataSource`, `FeatureTransformer`, `VolatilityModel`, `Metric`, `ArtifactStore`, `ReportRenderer`  
- **errors:** typed failures (`DataValidationError`, `LeakageError`, `ConfigError`)  
- No pandas/sklearn in protocols’ *conceptual* contracts where possible; adapters may accept DataFrames at boundaries

### Ingestion (`vip.ingestion`)

- Fetch OHLCV (and later options/VIX)  
- Normalize schema, adjust for corporate actions where the vendor supports it  
- Validate gaps, duplicates, timezone, survivorship assumptions  
- Persist raw + normalized tables via persistence layer

### Features (`vip.features`)

- **targets:** realized variance/vol at horizons h \in 1d, 5d, 21d; Parkinson, Garman–Klass, close-to-close  
- **predictors:** return lags, HAR components, ranges, volume shocks, IV proxies (if available), calendar, cross-asset  
- **registry:** name → builder for config-driven feature sets  
- **pipeline:** compose transformers with explicit lookback and shift rules to prevent leakage

### Modeling (`vip.modeling`)

- Baselines: historical mean, EWMA, classic HAR-RV (OLS)  
- Linear: Ridge / Lasso / Elastic Net  
- Nonlinear: Random Forest, LightGBM/XGBoost (optional)  
- Trainer: fit on train window only; serialize with metadata  
- Selection: nested or sequential walk-forward factor screening

### Evaluation (`vip.evaluation`)

- Loss functions suited to vol (QLIKE, MSE on RV or log-RV)  
- Time-series CV with purge/embargo  
- Walk-forward backtest engine  
- Factor importance + stability across regimes  
- Statistical comparison (M7: block bootstrap primary; optional HLN–DM + Newey–West)

### Visualization / Reporting

- Research-grade charts (not dashboard candy): RV vs forecast, rolling skill, importance heatmaps  
- HTML/Markdown experiment reports with config hash, data version, metrics table

### Persistence / Config / Orchestration / CLI

- Parquet + JSON/YAML metadata for reproducibility  
- Pydantic-validated configs  
- DI container wires adapters  
- CLI mirrors the research workflow for demos and CI

---

## 4. Data Flow

```text
Config (symbol, dates, horizon, features, model)
        │
        ▼
┌───────────────┐     raw Parquet      ┌────────────────┐
│  Ingestion    │ ───────────────────► │ Persistence    │
└───────────────┘                      └───────┬────────┘
        │                                      │
        ▼                                      ▼
┌───────────────┐   aligned OHLCV      ┌────────────────┐
│  Validators   │ ◄────────────────────│  Load cache    │
└───────┬───────┘                      └────────────────┘
        ▼
┌───────────────┐
│ Feature pipe  │  target y_{t+h} from future RV; X_t from info ≤ t
└───────┬───────┘
        ▼
┌───────────────┐     FeatureMatrix + lineage
│  Persistence  │ ────────────────────────────────┐
└───────┬───────┘                                 │
        ▼                                         ▼
┌───────────────┐                         Experiment registry
│ Walk-forward  │◄──── split policy (train / embargo / test)
└───────┬───────┘
        │  per fold: fit model → predict → score
        ▼
┌───────────────┐
│  Evaluation   │ → metrics, importances, residuals
└───────┬───────┘
        ▼
┌───────────────┐     figures + HTML/MD
│ Viz + Report  │ ──────────────────────► artifacts/
└───────────────┘
```

**Leakage rules (non-negotiable):**

1. Target at t uses returns/RV strictly over (t, t+h].
2. Features at t use only data with timestamp \le t (end-of-day: close of t).
3. Scaling/feature selection fit only on the training segment of each fold.
4. Embargo ≥ feature lookback and ≥ horizon when labels overlap.

**Primary artifact lineage:** `config_hash` + `data_version` + `code_version` → every report can be reproduced.

---

## 5. External Libraries


| Area              | Library                                                    | Role                  |
| ----------------- | ---------------------------------------------------------- | --------------------- |
| Packaging         | `hatch` or `setuptools` + `pyproject.toml`                 | installable package   |
| Config            | `pydantic` v2, `pyyaml`                                    | validated settings    |
| Data              | `pandas`, `numpy`, `pyarrow`                               | frames + Parquet      |
| Market data       | `yfinance` (MVP); later `polygon-api-client` / `databento` | ingestion             |
| Calendar          | `exchange-calendars` or `pandas_market_calendars`          | sessions              |
| Stats / baselines | `statsmodels`                                              | HAR OLS, HAC SEs      |
| ML                | `scikit-learn`, optional `lightgbm`                        | models + pipelines    |
| Explainability    | `shap` (optional milestone)                                | nonlinear importance  |
| Viz               | `matplotlib`, `plotly`                                     | static + interactive  |
| Reporting         | `jinja2`, `markdown`                                       | HTML reports          |
| CLI               | `typer` or `click`                                         | operator interface    |
| Logging           | `structlog` or stdlib `logging`                            | structured logs       |
| Testing           | `pytest`, `pytest-cov`, `hypothesis`                       | unit + property tests |
| Quality           | `ruff`, `mypy`, `pre-commit`                               | lint/types            |
| Notebooks         | `jupyter`, `ipywidgets` (light)                            | research UI           |


**Avoid early:** full Airflow/Prefect, heavy FastAPI, cloud warehouses — add when the research loop is solid.

---

## 6. What Should Be Configurable

All of this belongs in YAML (and env for secrets), validated by Pydantic:


| Category       | Examples                                    |
| -------------- | ------------------------------------------- |
| **Universe**   | symbol(s), benchmark, optional VIX ticker   |
| **Sample**     | start/end, timezone, adjust prices          |
| **Target**     | RV estimator, horizon, annualization        |
| **Features**   | enabled groups, lookbacks, winsorization    |
| **Model**      | family, hyperparameters, random seed        |
| **Validation** | walk-forward window sizes, n_folds, embargo |
| **Metrics**    | primary metric (QLIKE), secondary list      |
| **Paths**      | raw/processed/artifact roots                |
| **Runtime**    | log level, n_jobs, cache refresh policy     |
| **Reporting**  | formats, figure theme, top-k factors        |


Experiment configs should be **diffable and checked into git**; secrets (API keys) only via environment.

---

## 7. Recommended Design Patterns


| Pattern                            | Where                                               | Why                                                     |
| ---------------------------------- | --------------------------------------------------- | ------------------------------------------------------- |
| **Protocol / Strategy**            | data sources, models, metrics, feature builders     | swap Yahoo↔Polygon, HAR↔GBM without rewriting pipelines |
| **Registry**                       | features, models, metrics                           | config strings → implementations                        |
| **Pipeline**                       | feature + sklearn-style transforms                  | composable, testable stages                             |
| **Factory**                        | model/feature construction from config              | single place for wiring                                 |
| **Dependency Injection**           | `orchestration/container.py`                        | test doubles, no global state                           |
| **Template Method**                | walk-forward runner                                 | fixed fold loop; pluggable fit/predict/score            |
| **Adapter**                        | yfinance → domain bars; sklearn → `VolatilityModel` | isolate vendor APIs                                     |
| **Value Object**                   | `DateRange`, `Horizon`, `ExperimentId`              | invalidate illegal states early                         |
| **Unit of Work / Artifact bundle** | persistence of one experiment                       | atomic metadata + predictions + figures                 |
| **Builder** (optional)             | complex experiment specs                            | readable construction in notebooks                      |


**SOLID mapping (pragmatic):**

- **S:** one module owns RV targets; one owns walk-forward  
- **O:** new features/models via registry, not edits to the runner  
- **L/I:** small protocols (`fit`/`predict`, not god-interfaces)  
- **D:** application depends on `MarketDataSource`, not `YFinanceSource`

---

## 8. Development Roadmap

### Milestone 0 — Foundations (week 1) - DONE (2026-07-28)

Completed:

- Package skeleton, `pyproject.toml`, lint/type/test CI  
- Domain entities + protocols  
- Config schema + `default.yaml`  
- Parquet store + logging  
**Exit:** `pip install -e .` and empty CLI `vip --help`

### Milestone 1 — Data spine - DONE (2026-07-28)

Completed:

- yfinance adapter + OHLCV validators
- ingest use-case + `vip ingest`
- SPY Parquet persistence under `data/raw/`
- Unit tests (network-free) and package docs

### Milestone 2 — Features & targets - DONE (2026-07-28)

Completed:

- RV targets (multi-horizon) + HAR / return / range / volume features  
- Feature registry + leakage unit tests (shift/alignment assertions)  
**Exit:** feature matrix for SPY with documented column dictionary

### Milestone 3 — Baselines & evaluation — DONE (2026-07-28)

Completed:

- Historical mean, EWMA, HAR-RV OLS baselines
- QLIKE/MSE/MAE + expanding walk-forward with embargo
- `vip evaluate` comparison table + artifact persistence

### Milestone 4 — Factor intelligence — DONE (2026-07-28)

Completed:

- Regularized linear models (Ridge/Lasso/ElasticNet) + permutation importance
- Factor screening use-case, stability ranking, importance plot
- First HTML research report via `vip screen`

### Milestone 5 — Nonlinear & robustness — DONE (2026-07-29)

Completed:

- RandomForest tree model + registry wiring
- Median/capped permutation importance aggregation
- VIX cross-asset features with backward asof join + leakage tests
- Regime-sliced OOS metrics (COVID crash, 2022 bear, full sample)
- Optional TreeSHAP attribution (behind `[nonlinear]` extra)
- "What works when" section in HTML report
- Multi-symbol batch screening (`vip screen-batch`)

### Milestone 6 — Platform polish (portfolio-ready) - DONE (2026-07-29)

- Full CLI: ingest → features → experiment → report  
- Integration tests + golden-file metrics for one frozen config  
- Docs: methodology, architecture, how to add a feature  
- Optional: thin FastAPI `GET /experiments/{id}` for demo  
**Exit:** one-command reproduction of the flagship SPY/QQQ study

### Milestone 7 — Statistical inference on OOS gaps - done (2026-08-01)

**Motivation.** Mean OOS QLIKE rankings (e.g. Lasso vs HAR) are descriptive.
Overlapping multi-day RV labels inflate effective dependence, so a gap such as
~0.05 QLIKE is an observation until we report uncertainty. Embargo prevents
train/test leakage; it does not establish that a model gap is statistically real.

**Scope:**

- Persist per-observation (or per-row) OOS losses alongside fold aggregates
- HAC / Newey–West lag locked to `horizon_days - 1` (default **4** for the 5-day target)
- Primary inference: **block bootstrap** of mean OOS ΔQLIKE vs baseline
(`har_rv_ols` by default; block length 10–20 trading days; default **15**),
reporting mean gap, bootstrap CI, and bootstrap p-value — preferred because
QLIKE loss differentials are asymmetric and fold-count is small (~5); plain
DM on few windows tends to over-reject
- Secondary (optional): Diebold–Mariano with Newey–West + **Harvey–Leybourne–Newbold**
finite-sample correction (never ship uncorrected DM as the sole claim)
- Report for each horse-race pair vs baseline: mean ΔQLIKE, bootstrap CI /
p-value (primary); if enabled, HLN–DM statistic and p-value (secondary)
- Optional sensitivity: non-overlapping evaluation subsample (every `horizon`
days) as a footnote check
- Wire results into `metrics.json` / comparison table and the HTML research memo
- Tighten report wording: “lower mean OOS QLIKE” vs “significantly better”
only when the **primary** (bootstrap) test rejects at configured α (default 0.05)
- Unit tests on synthetic loss differentials (known mean / known null)
- Document overlap, effective sample size, block bootstrap, NW lag, and optional
HLN–DM in `docs/research_methodology.md`

**Exit:** Flagship SPY (and batch) reports show model gaps with bootstrap CI /
p-values (HLN–DM optional alongside); methodology states that rankings without
inference are not findings.


### Milestone 8 — Multi-horizon factor intelligence — DONE (2026-08-04)

**Motivation.** Horizon was a single `target.horizon_days` knob. PMs need one
study that answers what predicts next-day vs next-week vs next-month RV under
the same horse-race and M7 inference contract.

**Scope (locked):**

- Orchestrate screens over horizons **{1, 5, 21}** via `screen_multi_horizon`
  / `vip screen-horizons`
- Per horizon: `target_rv_cc_{h}d`, `embargo_size = h`, `nw_lags = h − 1`,
  block bootstrap primary vs `har_rv_ols` (block defaults **10 / 15 / 21**;
  ranges 5–15 / 10–20 / 15–42)
- Cross-horizon `horizon_summary.json` + HTML **Skill by horizon**
- Keep single-horizon `vip screen` / `vip run` (default h=5) backward compatible
- Methodology + package docs; full pytest green for exit
- Stretch: daily jump-robust feature family (registry + leakage tests);
  CLI opt-in via ``--with jump`` / ``--with vix,jump``

**Exit checklist (code vs docs):**

- [x] Horizon default helpers + unit tests (`horizon_defaults`,
  `test_multi_horizon_defaults.py`)
- [x] Single-horizon screen injectable for h∈{1,5,21}
  (`settings_for_horizon`, `test_screen_factors_horizon.py`)
- [x] `screen_multi_horizon` writes `h{h}d/` + study root
- [x] M7 inference wired per horizon
- [x] `horizon_summary.json` + HTML “Skill by horizon”
- [x] CLI `vip screen-horizons`
- [ ] Methodology § Multi-horizon evaluation + README alignment (Agent E)
- [ ] Full `pytest -q` green after docs; then flip status to DONE
- [x] Stretch: jump family + leakage tests + CLI ``--with jump``
  (``features`` / ``run`` / ``screen-horizons``)

**Missing for DONE:** apply methodology/README/`plan` packaging below; run
flagship SPY sequence; confirm `pytest -q`. Then change heading to
`DONE (YYYY-MM-DD)` and tick the remaining boxes.


### Later — post-M7 research backlog (ordered; not committed)

Stochastic calculus and richer data are **enhancements**, not a second product.
New work must still plug into registries, walk-forward QLIKE evaluation, and
leakage tests. Prefer physical-measure RV research over a bolted-on options
pricing lab.

**Near-term extensions (fit the current spine)**

- *(Milestone 8)* Multi-horizon screens and stretch daily jump proxies — see
  Milestone 8 section above (core code in tree; exit gated on docs + pytest)
- Additional cross-asset covariates behind `MarketDataSource` + feature
registry (e.g. Treasury yields, simple sector/ETF returns) with as-of joins
- Parametric / filter baselines in the same horse-race (e.g. discrete OU-style
vol mean reversion, simple stochastic-vol inspired filters) — must beat HAR
on OOS QLIKE to matter
- Stronger realized-vs-implied studies using existing VIX (and later single-name
IV when a vendor exists): IV level, IV−RV gap as features; IV as a competing
forecast of forward RV

**Optional research diagnostics (secondary to OOS skill)**

- Granger causality screens for selected feature → forward-RV pairs
- Mutual information vs linear correlation as dependence diagnostics
- Event studies (e.g. pre/post earnings) once an earnings calendar source exists
- Monte Carlo scenario evaluation *from* walk-forward RV forecasts (path bands /
distributional metrics) — evaluation appendix, not the core claim
- Rough-volatility–inspired features (e.g. log-vol memory) as an advanced
optional family

**Explicitly deferred (higher data/ops cost)**

- Intraday / high-frequency RV
- Options-implied surfaces and single-name IV vendors (Polygon / similar)
- Option Greeks, variance-reduced MC pricing, Malliavin-based Greek estimation
(only reconsider after options data is in scope and IV−RV research is real)
- News / social sentiment pipelines
- Portfolio-of-names / cross-sectional models
- Live scheduling / production monitoring

---

## Design Principles to Lock In Now

1. **Research reproducibility over model novelty** — every claim ties to config + data version.
2. **Baselines first** — sophisticated models must beat HAR-RV in walk-forward.
3. **Leakage as a first-class test suite** — not a README warning.
4. **Config-driven experiments** — notebooks explore; CLI/configs are source of truth.
5. **Grow behind registries** — new factors/models shouldn’t touch the orchestrator.

---

## Locked Decisions

- Import package: vip
- Distribution name: volatility-intelligence-platform
- MVP vendor: yfinance (MarketDataSource protocol; no second vendor in MVP)
- Flagship symbol (M1): SPY
- Multi-symbol: design paths for it; implement batch in M4/M5
- Primary horizon: 5 trading days
- Primary target: close-to-close realized volatility
- Primary metric: QLIKE (secondary: MSE, MAE)
- M7 inference: block bootstrap primary (block length default 15); NW lags = horizon−1; HLN–DM secondary
- M8 horizons: 1 / 5 / 21; embargo = h; nw_lags = h−1; bootstrap ℓ defaults 10 / 15 / 21
- M8 CLI: `vip screen-horizons`; single-horizon default remains 5-day

---

## Status

### Milestone 0 — Foundations — DONE (2026-07-28)

### Milestone 1 — Data spine — DONE (2026-07-28)

### Milestone 2 — Features & targets — DONE (2026-07-28)

### Milestone 3 — Baselines & evaluation — DONE (2026-07-28)

### Milestone 4 — Factor intelligence — DONE (2026-07-28)

### Milestone 5 — Nonlinear & robustness — DONE (2026-07-29)

### Milestone 6 — Platform polish (portfolio-ready) - DONE (2026-07-29)

### Milestone 7 — Statistical inference on OOS gaps — DONE (2026-08-01)

### Milestone 8 — Multi-horizon factor intelligence — DONE (2026-08-04)

Code in tree (orchestration, horizon defaults, CLI, HTML, unit tests). Exit
blocked on methodology multi-horizon section + green full pytest; stretch CLI
jump flag optional.

Next after M8 DONE: remaining post-M7 backlog (cross-asset / IV−RV, parametric
vol baselines, optional diagnostics); keep HF RV, options surfaces, cross-section,
and scheduling deferred.


## Suggested Flagship Demo Narrative

> For liquid US ETFs (SPY, QQQ, IWM), screen which feature families predict next-5-day realized volatility; show that HAR components dominate short horizons while volume/range and VIX add incremental QLIKE skill in stress regimes; ship an HTML research memo a PM could skim.

---

