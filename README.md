# ContextGate

**RAG QA Gate for RAG engineers: grounded answers only.**

ContextGate decides whether a knowledge-base answer is allowed. It retrieves evidence, scores
whether that evidence is sufficient, validates citations and claims, then either answers with
traceable sources or abstains with a machine-readable reason.

```text
Connect documents
-> Ask a question
-> Retrieve evidence
-> Gate generation
-> Answer with citations or abstain
-> Inspect trace, cost and failure reason
-> Evaluate and promote policy
```

ContextGate is not a chatbot template. It is API-first infrastructure for teams building
production-ish RAG systems that need answer admission control, traceable citations, cost/latency
metadata, OpenAI-compatible serving, MLflow evaluation, and policy promotion evidence.

## Why Not Just Ask Claude Or OpenAI?

A single LLM request can answer one question over copied context. It does not give a production team
a repeatable gate for deciding whether the answer should be allowed.

ContextGate is useful when you need to prove and automate the parts around the prompt:

- stable answer vs abstain decisions;
- machine-readable abstention reasons such as `retrieval_empty` or `low_coverage`;
- citation and unsupported-claim checks before returning an answer;
- run traces, selected policy/provider, latency and cost records;
- gateway-level evaluation reports that show false answers, false abstentions and regressions;
- OpenAI-compatible serving without hiding grounding metadata.

The core value is not "RAG can answer from documents." The core value is **RAG QA control**:
answer only when the system can show why the answer is grounded.

## Who Should Use This?

Use ContextGate if you are:

- a RAG engineer comparing retrieval policies before shipping an answer API;
- an AI engineer who needs answer/citation traces, not just final text;
- a backend engineer adding production-shaped jobs, API keys, rate limits and metrics around RAG;
- a team that wants OpenAI-compatible chat endpoints backed by your own documents and evidence
  gates.

Do not use ContextGate if you need:

- a drag-and-drop chatbot builder;
- a CMS or document authoring system;
- OCR, image/table extraction or scanned PDF parsing in v0.1;
- multi-tenant RBAC in v0.1;
- a full prompt-injection security platform.

## 5-Minute Demo

Requirements: Docker Compose. The demo works without GPU and without a paid LLM; it can run with the
extractive fallback provider.

```bash
cp .env.example .env
docker compose up --build -d postgres redis qdrant mlflow api
docker compose exec api ctxgate ingest demo/documents --knowledge-base demo
```

This loads 27 multilingual Markdown policy documents into the `demo` knowledge base. The first run
downloads embedding models into the Docker `model-cache` volume; later runs are faster.

Services:

| Service | URL |
|---|---|
| FastAPI/OpenAPI | http://localhost:8000/docs |
| Chainlit operator console | http://localhost:8001 |
| MLflow | http://localhost:5000 |
| Qdrant | http://localhost:6333/dashboard |

Development API key: `contextgate-dev-key`.

Quick test questions:

- grounded: `How can I cancel an order?`
- abstention: `Do you accept cryptocurrency?`

Full QA gate benchmark/router demo:

```bash
docker compose exec api ctxgate demo
```

The demo prints one grounded answer, one structured abstention, and a QA Gate report path. The HTML
report starts with answer rate, abstention rate, false answer rate, false abstention rate, citation
validity and the concrete failed cases.

For local Python development:

```bash
uv sync --extra dev --extra ui --extra llm
uv run ctxgate demo
uv run ctxgate serve
```

Fast deterministic smoke mode without model downloads:

```bash
CONTEXTGATE_EMBEDDING_BACKEND=deterministic \
CONTEXTGATE_DENSE_DIMENSION=64 \
CONTEXTGATE_LATE_DIMENSION=32 \
uv run ctxgate demo
```

## Ask A Grounded Question

Native API:

```bash
curl http://localhost:8000/api/v1/runs/answer \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_base": "demo",
    "query": "How can I cancel an order?",
    "policy": "auto",
    "latency_budget_ms": 800,
    "cost_budget_usd": 0.002,
    "debug": true
  }'
```

Expected response shape:

```json
{
  "answer": "An order can be cancelled before it is handed to the courier. [1]",
  "citations": [{"index": 1, "chunk_id": "cancel-order-en:0", "source": "cancel-order-en.md"}],
  "provider": "extractive",
  "selected_provider": "extractive",
  "grounded": true,
  "evidence_score": 0.82,
  "abstention_reason": null,
  "retrieval": {
    "policy": "balanced",
    "abstained": false,
    "trace_id": "..."
  },
  "cost": {"estimated_usd": 0.0}
}
```

When evidence is insufficient:

```json
{
  "answer": "I could not answer from grounded evidence in the knowledge base. Abstention reason: retrieval_empty.",
  "citations": [],
  "provider": "abstention",
  "selected_provider": "abstention",
  "grounded": false,
  "evidence_score": 0.0,
  "abstention_reason": "retrieval_empty",
  "retrieval": {
    "abstained": true,
    "trace_id": "..."
  }
}
```

OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kb:demo:auto",
    "messages": [{"role": "user", "content": "How can I cancel an order?"}]
  }'
```

OpenAI responses include a `contextgate` metadata block:

```json
{
  "contextgate": {
    "run_id": "...",
    "trace_id": "...",
    "selected_retrieval_policy": "balanced",
    "selected_provider": "extractive",
    "evidence_score": 0.82,
    "abstained": false,
    "abstention_reason": null,
    "grounded": true,
    "cost": {"estimated_usd": 0.0},
    "citations": []
  }
}
```

Inspect trace and cost:

```bash
curl http://localhost:8000/api/v1/runs/{run_id}/trace -H "X-API-Key: contextgate-dev-key"
curl http://localhost:8000/api/v1/runs/{run_id}/cost  -H "X-API-Key: contextgate-dev-key"
```

## What It Does

- Ingests PDF, Markdown, HTML and TXT into managed Qdrant collections.
- Runs dense, sparse/BM25, hybrid/RRF and ColBERT late-interaction retrieval policies.
- Uses LangGraph for normalize -> retrieve -> score evidence -> generate/abstain -> verify citations.
- Serves native `/api/v1/*` APIs and OpenAI-compatible `/v1/chat/completions`.
- Persists durable jobs, policies, traces, run events and cost records.
- Logs MLflow experiments, router artifacts, dataset fingerprints and QA Gate HTML reports.

## Core Concepts

- **Evidence score**: combines answerability, coverage and support.
- **Abstention**: stable machine-readable refusal when retrieval/evidence/citations fail.
- **Citation validation**: citations must reference retrieved chunks and valid ranks.
- **Retrieval policy**: `fast`, `balanced`, `accurate`, or `auto`.
- **Provider routing**: extractive fallback by default; LiteLLM/OpenAI-compatible provider when
  configured.
- **Trace**: node-level record of query analysis, retrieval hits, evidence scoring, provider choice,
  citations and final result.

See [docs/concepts.md](docs/concepts.md) for details.

## Workflows

CLI:

```bash
ctxgate demo
ctxgate ingest ./documents --knowledge-base support
ctxgate sync-qdrant source_collection --knowledge-base support
ctxgate benchmark ./evaluation.jsonl --knowledge-base support --evaluate-answers
ctxgate router train BENCHMARK_RUN_ID --knowledge-base support
ctxgate router promote RUN_ID --knowledge-base support
ctxgate serve
```

Native API groups:

```http
POST /api/v1/knowledge-bases
POST /api/v1/knowledge-bases/{id}/documents
POST /api/v1/retrieve
POST /api/v1/runs/answer
GET  /api/v1/runs/{run_id}/trace
GET  /api/v1/runs/{run_id}/cost
POST /api/v1/evaluations
POST /api/v1/routers/train
POST /api/v1/routers/promote
GET  /api/v1/providers
```

With `--evaluate-answers`, the benchmark evaluates the full answer gateway rather than only
retrieval. The report shows which answerable queries were abstained, which unanswerable queries were
answered, and which grounded answers failed citation/fact checks.

See [docs/workflows.md](docs/workflows.md) for copy-paste workflows.

## Architecture

Dependency rule:

```text
domain -> application -> ports -> adapters -> apps
```

- `domain`: evidence, citations, retrieval results, errors and pure policies.
- `application`: use cases such as answer, retrieve, ingest, evaluate, train and promote.
- `ports`: vector index, router, providers, jobs, traces, cost ledger and cache contracts.
- `adapters`: Qdrant, FastEmbed, LangGraph, LiteLLM, MLflow, SQLAlchemy, Celery and Redis.
- `apps`: FastAPI, CLI, worker, Chainlit and MLflow entrypoints.

See [docs/architecture.md](docs/architecture.md).

## Configuration

`docker-compose.yml` reads `.env` and passes configuration components into the app. ContextGate
`Settings` builds database, Redis, Qdrant and MLflow URLs inside Python instead of hardcoding DSNs in
Compose.

Key local settings:

- `CONTEXTGATE_API_KEY`
- `CONTEXTGATE_DATABASE_NAME`, `CONTEXTGATE_DATABASE_USER`, `CONTEXTGATE_DATABASE_PASSWORD`
- `CONTEXTGATE_REDIS_PASSWORD`
- `CONTEXTGATE_LLM_MODEL`, `CONTEXTGATE_LLM_API_BASE`, `CONTEXTGATE_LLM_API_KEY`
- `CONTEXTGATE_API_PORT`, `CONTEXTGATE_POSTGRES_PORT`, `CONTEXTGATE_REDIS_PORT`,
  `CONTEXTGATE_QDRANT_PORT`, `CONTEXTGATE_MLFLOW_PORT`, `CONTEXTGATE_CHAINLIT_PORT`
- `CONTEXTGATE_*_SERVICE` only when changing Docker Compose internal service names
- `CONTEXTGATE_MLFLOW_WORKERS`, `CONTEXTGATE_MLFLOW_ALLOWED_HOSTS` and
  `CONTEXTGATE_MLFLOW_CORS_ALLOWED_ORIGINS` for MLflow startup/security tuning

Database migrations live beside SQLAlchemy models:

```bash
uv run alembic -c src/contextgate/adapters/sqlalchemy/alembic.ini upgrade head
```

## Documentation

- [Product](docs/product.md): audience, use cases, non-goals and project promise.
- [Concepts](docs/concepts.md): evidence gate, retrieval policies, citations, traces and routing.
- [Workflows](docs/workflows.md): ingest, answer, inspect, evaluate and promote.
- [Architecture](docs/architecture.md): clean architecture, runtime, jobs and adapters.
- [Demo Quickstart](demo/README.md): load demo data, ask grounded/unanswerable questions.
- [Demo Dataset](demo/DATASET.md): multilingual benchmark dataset details.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/contextgate
uv run deptry .
uv audit --locked
uv run pytest --cov=contextgate --cov-report=term-missing
uv run alembic -c src/contextgate/adapters/sqlalchemy/alembic.ini upgrade head
uv run alembic -c src/contextgate/adapters/sqlalchemy/alembic.ini check
docker compose config --quiet
docker build -t contextgate:local .
```

Smoke:

```bash
cp .env.example .env
docker compose up -d postgres redis qdrant mlflow api
curl http://127.0.0.1:8000/health
```

## Status

`v0.1.0-alpha`. Single-tenant by design for now. API keys, rate limits, idempotency, durable jobs,
traces, cost ledger, typed errors and observability are implemented in a production-shaped way.

## License

Apache-2.0. See [LICENSE](LICENSE).
