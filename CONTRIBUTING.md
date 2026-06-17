# Contributing

## Development setup

```bash
uv sync --extra dev --extra ui --extra llm
uv run pytest
uv run ruff check .
uv run mypy src/ContextGate
```

Use Python 3.12. Do not commit downloaded models, local databases, reports, or customer data.

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

