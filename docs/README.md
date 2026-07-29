# Documentation Index

This folder collects project-level docs for the Volatility Intelligence Platform.

## Start here

- Architecture and roadmap: [`../plan.md`](../plan.md)
- Research methodology: [`research_methodology.md`](research_methodology.md)
- Contributing and local setup: [`contributing.md`](contributing.md)
- Package overview: [`../README.md`](../README.md)
- Milestone walkthroughs: [`../milestone walkthroughs/`](../milestone%20walkthroughs/)

## Package docs (`src/vip`)

- [`domain`](../src/vip/domain/README.md) — entities, value objects, protocols, errors
- [`config`](../src/vip/config/README.md) — YAML loading and validated settings
- [`persistence`](../src/vip/persistence/README.md) — Parquet market data, feature matrices, and artifacts
- [`orchestration`](../src/vip/orchestration/README.md) — logging and (later) wiring
- [`cli`](../src/vip/cli/README.md) — terminal commands (`info`, `ingest`, `features`, `evaluate`, `screen`)
- [`ingestion`](../src/vip/ingestion/README.md) — Yahoo Finance adapter and OHLCV validation
- [`features`](../src/vip/features/README.md) — targets, feature families, registry, and pipeline
- [`modeling`](../src/vip/modeling/README.md) — baselines and regularized linear models
- [`evaluation`](../src/vip/evaluation/README.md) — metrics, walk-forward, importance, stability
- [`visualization`](../src/vip/visualization/README.md) — research plots (importance bars)
- [`reporting`](../src/vip/reporting/README.md) — HTML experiment memos
- [`application`](../src/vip/application/README.md) — use-cases for ingest, features, baselines, screening

## Planned docs

As later milestones land, add:

- `architecture.md` — deeper design notes beyond `plan.md`

## Conventions

- Code docstrings use NumPy style.
- Package READMEs stay short and practical (purpose, APIs, boundaries).
- Research claims should be reproducible from config + code + data version.