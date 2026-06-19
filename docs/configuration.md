# Configuration

ContextGate reads `CONTEXTGATE_*` environment variables. `get_settings()` loads `.env` by default;
set `CONTEXTGATE_ENV_FILE` to another path or an empty value to disable dotenv loading.

## Runtime And Security

| Variable | Default | Production guidance |
|---|---|---|
| `ENVIRONMENT` | `development` | Set `production`; enables secret validation |
| `AUTH_ENABLED` | `false` in Python, `true` in Compose | Must be true |
| `API_KEY` | `contextgate-dev-key` | Replace with a high-entropy bootstrap key |
| `RATE_LIMIT_ENABLED` | `true` | Keep enabled unless enforced upstream |
| `RATE_LIMIT_PER_MINUTE` | `120` | Tune from measured traffic |
| `RATE_LIMIT_FAIL_OPEN` | `true` in Python, `false` in Compose | Keep false |
| `GRAPH_CHECKPOINTING_ENABLED` | `false` | Enable only for graph-resume development; durable run traces do not require it |
| `TRACE_CONTENT_MODE` | `full` | Use `metadata` for sensitive data |
| `TRACE_RETENTION_DAYS` | `30` | Match privacy and audit policy |

## PostgreSQL, Redis, And Qdrant

Database settings are `DATABASE_URL` or `DATABASE_BACKEND`, `DATABASE_DRIVER`, `DATABASE_HOST`,
`DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USER`, and `DATABASE_PASSWORD`. Redis uses `REDIS_URL`
or `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, and `REDIS_PASSWORD`. Qdrant uses `QDRANT_URL` or
`QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`, and `QDRANT_LOCAL_PATH`.

The API SQL pool defaults to `DATABASE_POOL_SIZE=10`, `DATABASE_MAX_OVERFLOW=5`, and
`DATABASE_POOL_TIMEOUT_SECONDS=10`. Ensure PostgreSQL `max_connections` covers the sum across all
API replicas and workers before scaling horizontally.

Compose's demo configuration uses `API_WORKERS=4` with deterministic embeddings. Start with one or
two workers when each process loads FastEmbed models, measure resident memory, then increase only
within the host's memory and PostgreSQL connection budgets.

Compose service-name overrides are `POSTGRES_SERVICE`, `REDIS_SERVICE`, `QDRANT_SERVICE`,
`API_SERVICE`, and `MLFLOW_SERVICE`. Host exposure is controlled by `HOST_BIND_ADDRESS` and each
`*_PORT` variable.

`API_PUBLIC_URL` and `MLFLOW_PUBLIC_URL` control host-reachable links printed by CLI workflows.
They default to `http://localhost:8000` and `http://localhost:5000`; set them to the TLS proxy URLs
used by remote deployments.

## Retrieval And Storage

| Variable | Default | Meaning |
|---|---:|---|
| `EMBEDDING_BACKEND` | `fastembed` in Python, `deterministic` in the demo | Use FastEmbed for production-quality semantic retrieval |
| `DENSE_MODEL` / `DENSE_DIMENSION` | multilingual MiniLM / 384 | Dense embedding contract |
| `SPARSE_MODEL` | `Qdrant/bm25` | Sparse model |
| `LATE_MODEL` / `LATE_DIMENSION` | AnswerAI ColBERT / 96 | Late interaction contract |
| `LATE_INTERACTION_LANGUAGES` | `en` | Comma-separated languages or `*` |
| `QDRANT_STRICT_MODE` | `true` | Require indexed filter fields |
| `INDEXED_METADATA_FIELDS` | `{}` | JSON field-to-type map |
| `POLICIES_PATH` | `configs/policies.yaml` | Retrieval policy definitions |
| `UPLOAD_DIR` | `data/uploads` | Managed upload root |
| `REPORT_DIR` | `reports` | Evaluation report root |
| `ROUTER_DIR` | `data/routers` | Router artifact root |
| `MAX_UPLOAD_BYTES` | 20 MiB | Request upload limit |
| `MAX_PDF_PAGES` | 500 | PDF parser page limit |
| `MAX_EXTRACTED_CHARS` | 10,000,000 | Post-parser text limit |

Embedding dimensions are properties of models, not uploaded documents. FastEmbed validates that
`DENSE_DIMENSION` and `LATE_DIMENSION` equal the output sizes declared by `DENSE_MODEL` and
`LATE_MODEL`. Qdrant collection dimensions are immutable: after changing either model contract,
create a new knowledge base/collection and re-ingest. Never reset production volumes to change a
model. `PIPELINE_VERSION` is limited to 32 characters.

Compose uses named `DATA_VOLUME`, `REPORTS_VOLUME`, `model-cache`, Qdrant, Redis, and PostgreSQL
volumes by default. Set data/report volume variables to managed bind paths only when permissions and
backup behavior are understood.

## Generation And Budgets

Configure `LLM_MODEL`, `LLM_API_BASE`, and `LLM_API_KEY` for LiteLLM. Hard budgets additionally
require `LLM_INPUT_COST_PER_1M_TOKENS` and `LLM_OUTPUT_COST_PER_1M_TOKENS`. Other controls are
`LLM_MAX_OUTPUT_TOKENS`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`,
`LLM_CIRCUIT_FAILURE_THRESHOLD`, `LLM_CIRCUIT_COOLDOWN_SECONDS`, and
`ALLOW_PROVISIONAL_STREAMING`.

## Jobs, Evaluation, And MLflow

Worker controls are `WORKER_CONCURRENCY`, `WORKER_TASK_SOFT_TIME_LIMIT_SECONDS`, and
`WORKER_TASK_TIME_LIMIT_SECONDS`. Router promotion thresholds are configured with
`ROUTER_MIN_RELEASE_CASES`, `ROUTER_MIN_UNANSWERABLE_CASES`, `ROUTER_MIN_CASES_PER_LANGUAGE`,
`ROUTER_REQUIRED_LANGUAGES`, and the false-answer/citation/claim-support confidence bounds.

MLflow can use `MLFLOW_TRACKING_URI` locally or its service host/port, backend store, registry store,
artifact root, workers, allowed hosts, and CORS variables. It is not required for answer traffic.
