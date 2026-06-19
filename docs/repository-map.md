# Repository Map

## Top Level

| Path | Purpose |
|---|---|
| `README.md` | Product entry point and shortest demo |
| `AGENTS.md` and tool adapters | Canonical guidance for coding agents |
| `pyproject.toml` / `uv.lock` | Package metadata, dependencies, tools, locked versions |
| `Dockerfile` / `docker-compose.yml` | Application image and single-host topology |
| `.env.example` | Safe demo configuration and complete Compose inputs |
| `configs/policies.yaml` | Fixed retrieval policy parameters |
| `demo/` | Synthetic multilingual documents, benchmark, and dataset notes |
| `docker/` | Database initialization assets |
| `docs/` | Product, developer, API, operations, and release documentation |
| `scripts/` | Dataset generation, E2E, load, docs, and release-report utilities |
| `src/contextgate/` | Installable application source |
| `tests/` | Unit, contract, architecture, adapter, and security tests |
| `.github/` | CI, security scanning, and Copilot instructions |

Local `.contextgate/`, `data/`, `reports/`, `mlruns/`, caches, and virtual environments are runtime
artifacts and are ignored by Git.

## Source Package

| Directory/file group | Responsibility |
|---|---|
| `domain/` | Pure documents, retrieval, evidence, risk, evaluation, errors, and value objects |
| `application/dto.py` | Commands crossing into application use cases |
| `application/retrieval.py` | Retrieval orchestration independent of concrete storage |
| `application/use_cases.py` | Answer, ingestion, jobs, policy, key, trace, and promotion workflows |
| `ports/` | Protocols for repositories, vectors, routers, providers, and guardrails |
| `adapters/fastembed/` | Dense, sparse, late-interaction, and deterministic embeddings |
| `adapters/qdrant/` | Vector collection schema, indexing, search, probing, and filters |
| `adapters/langgraph/` | Admission graph and optional PostgreSQL checkpointing |
| `adapters/litellm/` | Provider selection, generation, streaming, usage, pricing, and circuit breaker |
| `adapters/local/` | File loaders, bounded parsing, ingestion, and local claim verification |
| `adapters/sqlalchemy/` | Models, repositories, ledger, unit of work, sessions, and Alembic migrations |
| `adapters/celery/` | Broker configuration, durable queues, tasks, and job runners |
| `adapters/mlflow/` | Gateway evaluation, reports, router training, checksums, and promotion gates |
| `apps/api/` | FastAPI routes, schemas, auth, scopes, rate limits, and OpenAI compatibility |
| `apps/cli/` | `ctxgate` initialization, doctor, demo, ingestion, evaluation, and serving commands |
| `apps/chainlit/` | Optional operator UI |
| `apps/worker/` | Celery worker entry point |
| `apps/mlflow/` | Optional MLflow service entry point |
| `apps/container.py` | Composition root and lifecycle/readiness ownership |
| `config/` | Typed settings and retrieval policy loading |
| `observability/` | Prometheus metric definitions |

Alembic files are ordered schema history and must not be rewritten after release. Tests mirror the
major runtime boundaries: graph/evidence/generation, retrieval/vector store, persistence/jobs,
API/auth/security, evaluation/router, and architectural dependency rules.
