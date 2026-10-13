# Contributing guide

## Setup

```bash
uv sync
```

## Tests

```bash
uv run pytest -q             # all tests (golden tests are marked slow, minutes here)
uv run pytest -q -m "not slow"  # fast lane
```

## Lint and formatting

Hooks are defined in `.pre-commit-config.yaml` and run with [prek](https://prek.j178.dev/):

```bash
PATH=/usr/bin:$PATH uvx prek run -a
```

`src/bonsai/_core/` and `tests/data/` are excluded: they must stay byte-identical to their
upstream/fixture source (see `docs/provenance.md`).

## Docs

```bash
uvx hatch run docs:build
```
