# Documentation Index

This folder collects project-level docs for the Volatility Intelligence Platform.

## Start here

- Architecture and roadmap: `../plan.md`](../[plan.md](http://plan.md))

- Contributing and local setup: `contributing.md`]([contributing.md](http://contributing.md))

- Package overview: `../README.md`](../[README.md](http://README.md))

## Package docs `src/vip`)

Each package has a short README describing purpose, modules, key APIs, and boundaries:

- `domain`](../src/vip/domain/[README.md](http://README.md)) — entities, value objects, protocols, errors

- `config`](../src/vip/config/[README.md](http://README.md)) — YAML loading and validated settings

- `persistence`](../src/vip/persistence/[README.md](http://README.md)) — Parquet market data and artifact storage

- `orchestration`](../src/vip/orchestration/[README.md](http://README.md)) — logging and (later) wiring

- `cli`](../src/vip/cli/[README.md](http://README.md)) — terminal commands

## Planned docs

As later milestones land, add:

- `research_methodology.md` — targets, metrics, walk-forward rules, leakage policy

- `architecture.md` — deeper design notes beyond `plan.md`

- Package READMEs for `ingestion`, `features`, `modeling`, `evaluation`, `visualization`, `reporting`

## Conventions

- Code docstrings use NumPy style.

- Package READMEs stay short and practical (purpose, APIs, boundaries).

- Research claims should be reproducible from config + code + data version.