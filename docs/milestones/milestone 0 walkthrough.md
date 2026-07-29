Milestone 0 is scaffolding only: installable package, thin domain/config/persistence/logging, empty CLI. No yfinance, features, or models yet.

Work in this order. After each step, stop and verify before moving on.

---

## Step 0 — Mental map of what exists at the end

```text
Volatility Intelligence Platform/
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   └── default.yaml
├── src/
│   └── vip/
│       ├── __init__.py
│       ├── py.typed
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── enums.py
│       │   ├── value_objects.py
│       │   ├── entities.py
│       │   ├── protocols.py
│       │   └── errors.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── defaults.py
│       │   └── loader.py
│       ├── persistence/
│       │   ├── __init__.py
│       │   ├── parquet_store.py
│       │   └── artifact_store.py
│       ├── orchestration/
│       │   ├── __init__.py
│       │   └── logging.py
│       └── cli/
│           ├── __init__.py
│           └── main.py
├── tests/
│   ├── unit/
│   │   ├── test_config_loader.py
│   │   ├── test_value_objects.py
│   │   └── test_parquet_store.py
│   └── test_cli.py
└── data/                    # created at runtime; gitignored
```

Skip CI (`.github/workflows/ci.yml`) until the package installs cleanly — add it as the last M0 file if you want.

---

## 1. `.gitignore`

**Purpose:** keep secrets, caches, and local data out of git.

**Include at least:**
- `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- `.venv/`, `venv/`
- `dist/`, `build/`, `*.egg-info/`
- `data/` (raw/processed/artifacts live here locally)
- `.env`
- IDE junk: `.idea/`, `.vscode/` (optional — some people commit VS Code settings)
- Jupyter: `.ipynb_checkpoints/`

**Verify:** file exists; nothing under `data/` will be committed later.

---

## 2. `pyproject.toml`

**Purpose:** define the installable project and tool config.

**Key fields:**
- `[project] name = "volatility-intelligence-platform"`
- `version = "0.1.0"`
- `requires-python = ">=3.11"` (3.11+ is a good bar)
- `dependencies` for M0 only:
  - `pandas`, `pyarrow`, `pydantic`, `pydantic-settings`, `pyyaml`, `typer`
- `[project.optional-dependencies] dev = ["pytest", "pytest-cov", "ruff", "mypy"]`
- `[project.scripts] vip = "vip.cli.main:app"` (Typer app object — see CLI section)
- `[build-system]` with `hatchling` (simplest modern default) **or** `setuptools`
- `[tool.hatch.build.sources] = ["src"]` / package under `src/vip` (exact syntax depends on hatch vs setuptools)
- `[tool.pytest.ini_options] testpaths = ["tests"]`
- `[tool.ruff]` / `[tool.mypy]` light defaults

**Important:** the **distribution name** is `volatility-intelligence-platform`; the **import package** is `vip` via the `src/` layout.

**Verify later:**  
`pip install -e ".[dev]"` then `python -c "import vip; print(vip.__version__)"`

---

## 3. `README.md`

**Purpose:** one-screen project pitch + how to install/run.

**Keep short for M0:**
- What the platform does (1 paragraph)
- Locked research defaults (SPY, 5d RV, QLIKE)
- Install: create venv → `pip install -e ".[dev]"`
- `vip --help`
- Point to `plan.md` for architecture

No need for full methodology yet.

---

## 4. Package root: `src/vip/__init__.py` and `src/vip/py.typed`

**`__init__.py`**
- Module docstring (PEP-257): one-line summary + list of top-level subpackages (`domain`, `config`, `persistence`, …).
- `__version__ = "0.1.0"` (match `pyproject.toml`).

**`py.typed`**
- Empty marker file so type checkers treat `vip` as typed.

**Verify:** `import vip` works after editable install.

---

## 5. Domain layer (create in this order)

### 5a. `src/vip/domain/errors.py`

Custom exceptions only — no logic.

Suggested:
- `VipError` (base)
- `ConfigError`
- `DataValidationError`
- `PersistenceError`

Each with a short docstring. Raise these later instead of bare `Exception`.

---

### 5b. `src/vip/domain/enums.py`

Small closed sets you’ll use in config and entities:

- `RvEstimator` — start with `CLOSE_TO_CLOSE` (add Parkinson etc. later)
- `MetricName` — `QLIKE`, `MSE`, `MAE`
- `PriceFrequency` — `DAILY` for now
- Maybe `SplitMode` — `WALK_FORWARD` as a placeholder

Use `enum.StrEnum` (3.11+) so YAML strings map cleanly.

---

### 5c. `src/vip/domain/value_objects.py`

Immutable validated types (frozen dataclasses or Pydantic models):

| Type | Responsibility |
|------|----------------|
| `DateRange` | `start`, `end`; reject `start > end` |
| `Horizon` | positive int trading days; default concept = 5 |
| `Symbol` | non-empty ticker string, normalize to upper |
| `ExperimentId` | opaque id string (uuid later); non-empty |

**Public methods:** e.g. `DateRange.contains(date)`, `Symbol.as_path_key()` → `"SPY"` for paths. Aim for ≥2 public methods per class (your Pylint rule).

**Verify with a unit test:** invalid `DateRange` / empty symbol raises.

---

### 5d. `src/vip/domain/entities.py`

Data shapes the rest of the system will pass around. For M0 keep them **thin**:

- `Bar` or document that OHLCV will be tabular at the boundary — many projects use a `MarketDataFrame` protocol later and keep entities as dataclasses for metadata:
  - `Instrument(symbol, currency="USD")`
  - `DatasetRef(symbol, path, content_hash optional)`
  - `ExperimentSpec` stub fields: symbol, horizon, estimator, primary_metric (or leave ExperimentSpec to config schema and keep entities minimal)

**M0 advice:** don’t over-model bars yet. Prefer:
- `Instrument`
- `DatasetRef`
- maybe `PredictionResult` as a stub with a docstring “filled in M3”

You’ll flesh entities when ingestion lands.

---

### 5e. `src/vip/domain/protocols.py`

**Typing.Protocol** interfaces — no implementations.

Minimum for M0 / near future:

```text
MarketDataSource
  - fetch(symbol, date_range) -> <tabular bars>   # return type: pandas.DataFrame OK at boundary for now
  - source_name -> str

FeatureBuilder
  - build(market_data) -> feature matrix
  - name -> str

VolatilityModel
  - fit(X, y) -> Self
  - predict(X) -> array-like

Metric
  - name -> str
  - compute(y_true, y_pred) -> float

ArtifactStore
  - write(experiment_id, name, payload)
  - read(...)

MarketDataStore   # parquet persistence of OHLCV
  - save(symbol, frame)
  - load(symbol) -> frame
  - exists(symbol) -> bool
```

Keep method counts small. Use `typing.Protocol` and `typing.Self` where useful.

Docstring each protocol: behavior + expected methods.

---

### 5f. `src/vip/domain/__init__.py`

Re-export the public surface you want: errors, key enums, value objects, protocols. Module docstring listing exports.

---

## 6. Config layer

### 6a. `configs/default.yaml`

Human-editable defaults matching locked decisions:

```yaml
symbol: SPY
date_range:
  start: "2015-01-01"
  end: null          # null = “through latest available” later
paths:
  raw_dir: data/raw
  processed_dir: data/processed
  artifacts_dir: data/artifacts
target:
  horizon_days: 5
  rv_estimator: close_to_close
evaluation:
  primary_metric: qlike
  secondary_metrics: [mse, mae]
logging:
  level: INFO
```

No model/feature blocks required yet — add empty placeholders if you want forward compatibility.

---

### 6b. `src/vip/config/schema.py`

Pydantic models mirroring the YAML (`BaseModel` or nested models). Validate enums via your `StrEnum`s.

Suggested top-level: `AppConfig` with nested `PathsConfig`, `TargetConfig`, `EvaluationConfig`, `LoggingConfig`.

**Constraint reminder:** keep constructors/functions ≤5 parameters; nest configs instead of flat kwargs.

---

### 6c. `src/vip/config/defaults.py`

Either:
- path constant to `configs/default.yaml` relative to repo root, or
- a function `default_config_path() -> Path`

Resolve paths relative to **project root** (directory containing `pyproject.toml` / `configs/`), not the cwd alone — document that assumption.

---

### 6d. `src/vip/config/loader.py`

Responsibilities:
1. Load YAML from a path.
2. Validate into `AppConfig`.
3. Raise `ConfigError` on missing file / bad schema (not bare `Exception`).

Public API sketch:
- `load_config(path: Path | None = None) -> AppConfig`
- `resolve_project_root() -> Path` (walk parents for `pyproject.toml`)

**Test:** load `configs/default.yaml` → assert `symbol == "SPY"`, `horizon_days == 5`.

---

## 7. Persistence (thin, real enough to test)

### 7a. `src/vip/persistence/parquet_store.py`

Implement something that satisfies the spirit of `MarketDataStore`:

- `__init__(self, root_dir: Path)`
- `symbol_path(self, symbol: str) -> Path` → e.g. `root_dir / symbol.upper() / "ohlcv.parquet"`
- `save(self, symbol: str, frame: pd.DataFrame) -> Path`
- `load(self, symbol: str) -> pd.DataFrame`
- `exists(self, symbol: str) -> bool`

Create parent dirs on save. Raise `PersistenceError` if load missing.

**Do not** call yfinance here — tests will pass a tiny synthetic DataFrame.

---

### 7b. `src/vip/persistence/artifact_store.py`

Generic experiment artifact writer:

- Root: `artifacts_dir / experiment_id /`
- `write_json`, `write_bytes`, or `write_frame` — keep to **2+ clear public methods**
- Used later for metrics JSON and reports; for M0, write/read a small JSON dict in a test

---

### 7c. `src/vip/persistence/__init__.py`

Export the two store classes. Docstring.

---

## 8. Logging: `src/vip/orchestration/logging.py`

**Purpose:** one place to configure logging so CLI and libraries don’t fight.

Suggested API:
- `configure_logging(level: str = "INFO") -> None`
- `get_logger(name: str) -> logging.Logger`

Use stdlib `logging` for M0 (structlog can wait). Idempotent configure (safe to call twice).

`orchestration/__init__.py` can be nearly empty with a package docstring. **Skip `container.py` until you have something to wire.**

---

## 9. CLI

### 9a. `src/vip/cli/main.py`

Typer app:

```text
app = typer.Typer(help="Volatility Intelligence Platform")

@app.callback()
def main(...):
    """Root callback; configure logging from options or defaults."""

@app.command("info")
def info():
    """Print package version and loaded default config summary."""
```

For M0 exit criterion `vip --help`, a Typer app with no subcommands is enough — but an `info` command that prints version + `symbol` / `horizon` from `load_config()` is a nice proof that config + package wiring work.

Wire entry point in `pyproject.toml` to this `app` object (Typer supports that).

### 9b. `src/vip/cli/__init__.py`

Short package docstring; usually no re-exports needed.

---

## 10. Tests (prove M0)

| File | What it locks in |
|------|------------------|
| `tests/unit/test_value_objects.py` | `DateRange` / `Symbol` validation |
| `tests/unit/test_config_loader.py` | `default.yaml` → `AppConfig` |
| `tests/unit/test_parquet_store.py` | save → exists → load roundtrip (tmp_path) |
| `tests/test_cli.py` | `CliRunner` → `vip --help` exit 0 |

Use pytest `tmp_path` for stores so you never write into real `data/` during tests.

---

## 11. Optional last: CI + empty data dirs

- `.github/workflows/ci.yml`: install `.[dev]`, run `ruff`, `pytest`
- Do **not** commit `data/`; `.gitignore` handles it. Stores create dirs on demand.

---

## Build sequence (checklist)

1. `.gitignore` + `pyproject.toml` + `README.md`
2. `src/vip/__init__.py` + `py.typed`
3. Domain: `errors` → `enums` → `value_objects` → `entities` → `protocols`
4. `configs/default.yaml` + config schema/loader
5. Persistence stores
6. Logging helper
7. CLI
8. `pip install -e ".[dev]"`
9. Tests + `vip --help`

**Milestone 0 exit criteria**
- [ ] `pip install -e ".[dev]"` succeeds  
- [ ] `import vip` works  
- [ ] `vip --help` works  
- [ ] `pytest` green (config, value objects, parquet roundtrip, CLI help)  
- [ ] No ingestion/ML code yet  

---

## How to use me while you build

Do **one step at a time**. Best message format:

> “I’ve created `pyproject.toml` and the `vip` package root. Here’s the content / error. What’s next?”

or

> “Review my `domain/protocols.py` before I write config.”

When you’re ready, start with **files 1–4** (gitignore, pyproject, README, package root), paste what you wrote or any install error, and we’ll tighten it before you touch domain code.