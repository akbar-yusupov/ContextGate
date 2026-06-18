# Contributing

## Development setup

```bash
uv sync --extra dev --extra ui --extra llm
uv run ruff check .
uv run ruff format --check .
uv run mypy src/contextgate
uv run deptry .
uv audit --locked
```

`deptry` rejects missing, transitive-only and unused direct dependencies. `uv audit` checks all
locked runtime, optional and development dependencies against OSV. `python-multipart` is the only
unused-import exception because FastAPI loads it as a runtime plugin for file uploads.

Use Python 3.12. Do not commit downloaded models, local databases, reports, or customer data.

## Running tests locally

Run the same test suite and coverage check as GitHub Actions before pushing:

```bash
uv sync --locked --extra dev
uv run pytest --cov=contextgate --cov-report=term-missing
```

The test configuration automatically uses deterministic embeddings, SQLite and local Qdrant
storage. PostgreSQL, Redis, Qdrant and MLflow services do not need to be running. To run a focused
test while developing, pass its path to pytest:

```bash
uv run pytest tests/test_router.py -q
```

## Pull requests

- Add focused tests for behavioral changes.
- Include benchmark evidence for retrieval or router changes.
- Report results for every affected language, including regressions.
- Document model names, dimensions, licenses, and hardware.
- Keep provider-specific code behind the existing embedding or generation interfaces.

## Benchmark contributions

A benchmark change must keep qrels deterministic and pass `tests/test_demo_dataset.py`. Synthetic
queries must be labeled as synthetic. Never contribute proprietary or personally identifiable
data.

## Design principle

ContextGate should remain a headless engineering tool with optional UI integrations. Avoid adding a
second document platform, chat product, or model proxy when a stable external tool can be
integrated instead.

