# Volatility intelligence platform

## User

I want to build a professional portfolio project called "Volatility Intelligence Platform".

The objective is to investigate which factors are most predictive of the future realized volatility of a single stock or ETF, and to build an end-to-end research platform suitable for demonstrating quantitative research, data science, and software engineering skills.

This project should be designed as if it were an internal research tool used by a quantitative hedge fund.

Requirements:

- Use Python.
- Organize the project as a proper Python package.
- Follow clean architecture and SOLID principles where appropriate.
- Separate data ingestion, feature engineering, modeling, evaluation, visualization, and reporting into independent modules.
- Prioritize readability, testability, and extensibility.
- Assume the project will eventually grow to 10,000+ lines of code.

Your first task is NOT to write code.

Instead:

1. Design the overall architecture.
2. Suggest the directory structure.
3. Identify every module and its responsibility.
4. Explain how data should flow through the system.
5. Identify external libraries that should be used.
6. Suggest which parts should be configurable.
7. Recommend design patterns where appropriate.
8. Produce a development roadmap divided into milestones.

Do not generate implementation code until the architecture has been finalized.

---

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

| Layer | Responsibility |
|--------|----------------|
| **Domain** | Types for bars, RV targets, feature sets, experiment specs; protocols for data sources, feature builders, models, metrics |
| **Application** | Use-cases: ingest → build features → train/evaluate → report; experiment runners |
| **Infrastructure** | Yahoo/Polygon clients, Parquet store, sklearn/LightGBM adapters, Plotly writers |
| **Interfaces** | CLI entrypoints, notebook helpers, config loading |

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
- **targets:** realized variance/vol at horizons \(h \in \{1d, 5d, 21d\}\); Parkinson, Garman–Klass, close-to-close  
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
- Statistical comparison (Diebold–Mariano later)  

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
1. Target at \(t\) uses returns/RV strictly over \((t, t+h]\).  
2. Features at \(t\) use only data with timestamp \(\le t\) (end-of-day: close of \(t\)).  
3. Scaling/feature selection fit only on the training segment of each fold.  
4. Embargo ≥ feature lookback and ≥ horizon when labels overlap.

**Primary artifact lineage:** `config_hash` + `data_version` + `code_version` → every report can be reproduced.

---

## 5. External Libraries

| Area | Library | Role |
|------|---------|------|
| Packaging | `hatch` or `setuptools` + `pyproject.toml` | installable package |
| Config | `pydantic` v2, `pyyaml` | validated settings |
| Data | `pandas`, `numpy`, `pyarrow` | frames + Parquet |
| Market data | `yfinance` (MVP); later `polygon-api-client` / `databento` | ingestion |
| Calendar | `exchange-calendars` or `pandas_market_calendars` | sessions |
| Stats / baselines | `statsmodels` | HAR OLS, HAC SEs |
| ML | `scikit-learn`, optional `lightgbm` | models + pipelines |
| Explainability | `shap` (optional milestone) | nonlinear importance |
| Viz | `matplotlib`, `plotly` | static + interactive |
| Reporting | `jinja2`, `markdown` | HTML reports |
| CLI | `typer` or `click` | operator interface |
| Logging | `structlog` or stdlib `logging` | structured logs |
| Testing | `pytest`, `pytest-cov`, `hypothesis` | unit + property tests |
| Quality | `ruff`, `mypy`, `pre-commit` | lint/types |
| Notebooks | `jupyter`, `ipywidgets` (light) | research UI |

**Avoid early:** full Airflow/Prefect, heavy FastAPI, cloud warehouses — add when the research loop is solid.

---

## 6. What Should Be Configurable

All of this belongs in YAML (and env for secrets), validated by Pydantic:

| Category | Examples |
|----------|----------|
| **Universe** | symbol(s), benchmark, optional VIX ticker |
| **Sample** | start/end, timezone, adjust prices |
| **Target** | RV estimator, horizon, annualization |
| **Features** | enabled groups, lookbacks, winsorization |
| **Model** | family, hyperparameters, random seed |
| **Validation** | walk-forward window sizes, n_folds, embargo |
| **Metrics** | primary metric (QLIKE), secondary list |
| **Paths** | raw/processed/artifact roots |
| **Runtime** | log level, n_jobs, cache refresh policy |
| **Reporting** | formats, figure theme, top-k factors |

Experiment configs should be **diffable and checked into git**; secrets (API keys) only via environment.

---

## 7. Recommended Design Patterns

| Pattern | Where | Why |
|---------|--------|-----|
| **Protocol / Strategy** | data sources, models, metrics, feature builders | swap Yahoo↔Polygon, HAR↔GBM without rewriting pipelines |
| **Registry** | features, models, metrics | config strings → implementations |
| **Pipeline** | feature + sklearn-style transforms | composable, testable stages |
| **Factory** | model/feature construction from config | single place for wiring |
| **Dependency Injection** | `orchestration/container.py` | test doubles, no global state |
| **Template Method** | walk-forward runner | fixed fold loop; pluggable fit/predict/score |
| **Adapter** | yfinance → domain bars; sklearn → `VolatilityModel` | isolate vendor APIs |
| **Value Object** | `DateRange`, `Horizon`, `ExperimentId` | invalidate illegal states early |
| **Unit of Work / Artifact bundle** | persistence of one experiment | atomic metadata + predictions + figures |
| **Builder** (optional) | complex experiment specs | readable construction in notebooks |

**SOLID mapping (pragmatic):**
- **S:** one module owns RV targets; one owns walk-forward  
- **O:** new features/models via registry, not edits to the runner  
- **L/I:** small protocols (`fit`/`predict`, not god-interfaces)  
- **D:** application depends on `MarketDataSource`, not `YFinanceSource`

---

## 8. Development Roadmap

### Milestone 0 — Foundations (week 1)
- Package skeleton, `pyproject.toml`, lint/type/test CI  
- Domain entities + protocols  
- Config schema + `default.yaml`  
- Parquet store + logging  
**Exit:** `pip install -e .` and empty CLI `vip --help`

### Milestone 1 — Data spine
- yfinance ingestion, validators, trading calendar alignment  
- Cache raw/normalized OHLCV  
- Smoke tests with recorded fixtures (no network in unit tests)  
**Exit:** `vip ingest --symbol SPY` writes reproducible Parquet

### Milestone 2 — Features & targets
- RV targets (multi-horizon) + HAR / return / range / volume features  
- Feature registry + leakage unit tests (shift/alignment assertions)  
**Exit:** feature matrix for SPY with documented column dictionary

### Milestone 3 — Baselines & evaluation
- HAR-RV OLS, EWMA, historical mean  
- QLIKE/MSE/MAE; walk-forward with embargo  
- Comparison tables  
**Exit:** notebook/CLI showing baselines beat naive mean on held-out folds

### Milestone 4 — Factor intelligence
- Regularized linear models + permutation importance  
- Factor screening use-case and stability plots  
- First HTML research report  
**Exit:** ranked factor list with confidence caveats (correlation, regime)

### Milestone 5 — Nonlinear & robustness
- LightGBM/RF optional path; SHAP summary  
- Cross-asset features (VIX); multi-symbol batch  
- Stress: COVID / 2022 regimes in evaluation slices  
**Exit:** “what works when” section in report

### Milestone 6 — Platform polish (portfolio-ready)
- Full CLI: ingest → features → experiment → report  
- Integration tests + golden-file metrics for one frozen config  
- Docs: methodology, architecture, how to add a feature  
- Optional: thin FastAPI `GET /experiments/{id}` for demo  
**Exit:** one-command reproduction of the flagship SPY/QQQ study

### Later (explicitly out of MVP)
- Intraday/high-frequency RV  
- Options-implied surfaces  
- Portfolio of names / cross-sectional models  
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

---

## Status

### Milestone 0 — Foundations — DONE (2026-07-28)

Completed:
- Installable package (`volatility-intelligence-platform` / import `vip`)
- Domain layer (errors, enums, value objects, entities, protocols)
- Config schema + `configs/default.yaml` + loader
- Parquet market-data store + filesystem artifact store
- Logging helpers + CLI (`vip --help`, `vip info`)
- Unit tests green; editable install works

Next: Milestone 1 — Data spine (yfinance ingest for SPY)

## Suggested Flagship Demo Narrative

> For liquid US ETFs (SPY, QQQ, IWM), screen which feature families predict next-5-day realized volatility; show that HAR components dominate short horizons while volume/range and VIX add incremental QLIKE skill in stress regimes; ship an HTML research memo a PM could skim.

---