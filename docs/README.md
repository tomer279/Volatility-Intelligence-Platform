# Documentation Index

This folder collects project-level docs for the Volatility Intelligence Platform.

## Start here

- Architecture and roadmap: [`../plan.md`](../plan.md)
- Contributing and local setup: [`contributing.md`](contributing.md)
- Package overview: [`../README.md`](../README.md)
- Milestone walkthroughs: [`../milestone walkthroughs/`](../milestone%20walkthroughs/)

## Package docs (`src/vip`)

- [`domain`](../src/vip/domain/README.md) — entities, value objects, protocols, errors
- [`config`](../src/vip/config/README.md) — YAML loading and validated settings
- [`persistence`](../src/vip/persistence/README.md) — Parquet market data, feature matrices, and artifacts
- [`orchestration`](../src/vip/orchestration/README.md) — logging and (later) wiring
- [`cli`](../src/vip/cli/README.md) — terminal commands (`info`, `ingest`, `features`, `evaluate`)
- [`ingestion`](../src/vip/ingestion/README.md) — Yahoo Finance adapter and OHLCV validation
- [`features`](../src/vip/features/README.md) — targets, feature families, registry, and pipeline
- [`modeling`](../src/vip/modeling/README.md) — volatility forecasting baselines
- [`evaluation`](../src/vip/evaluation/README.md) — metrics, walk-forward splits, comparison tables
- [`application`](../src/vip/application/README.md) — use-cases for ingest, features, and baseline evaluation

## Planned docs

As later milestones land, add:

- `research_methodology.md` — targets, metrics, walk-forward rules, leakage policy
- `architecture.md` — deeper design notes beyond `plan.md`
- Package READMEs for `visualization`, `reporting`

## Conventions

- Code docstrings use NumPy style.
- Package READMEs stay short and practical (purpose, APIs, boundaries).
- Research claims should be reproducible from config + code + data version.