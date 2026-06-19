# API Reference

All `/api/v1/*` and `/v1/*` requests require `X-API-Key` when authentication is enabled. The
interactive schema at `/docs` is authoritative for field validation and examples.

## Scopes

- `read`: list resources, jobs, runs, costs, evaluations, providers, and models.
- `write`: retrieve context and create answer/chat runs.
- `admin`: create knowledge bases, upload/sync data, manage jobs, policies, routers, evaluations,
  providers, and API keys. `admin` satisfies all scopes.

## Endpoints

| Method and path | Scope | Purpose |
|---|---|---|
| `GET /health` | public | Process liveness and version |
| `GET /ready` | public | PostgreSQL, Redis, and Qdrant readiness |
| `GET /metrics` | public/internal | Prometheus metrics; restrict at the reverse proxy |
| `POST /api/v1/knowledge-bases` | admin | Create a knowledge base |
| `GET /api/v1/knowledge-bases` | read | List knowledge bases |
| `GET /api/v1/knowledge-bases/{id}` | read | Read by UUID or slug |
| `GET /api/v1/knowledge-bases/{id}/documents` | read | List document versions |
| `POST /api/v1/knowledge-bases/{id}/documents` | admin | Upload and enqueue ingestion |
| `POST /api/v1/knowledge-bases/{id}/sync-qdrant` | admin | Copy an existing collection into managed storage |
| `GET /api/v1/jobs/{job_id}` | read | Poll job state/progress/result |
| `POST /api/v1/jobs/{job_id}/cancel` | admin | Request cooperative cancellation |
| `POST /api/v1/retrieve` | write | Retrieve without generation |
| `POST /api/v1/runs/answer` | write | Canonical evidence-gated answer endpoint |
| `GET /api/v1/runs/{run_id}` | read | Completed run summary |
| `GET /api/v1/runs/{run_id}/events` | read | Resumable SSE events |
| `GET /api/v1/runs/{run_id}/trace` | read | Completed JSON trace |
| `GET /api/v1/runs/{run_id}/cost` | read | Cost ledger records and total |
| `POST /api/v1/evaluations/datasets` | admin | Upload benchmark JSONL |
| `POST /api/v1/evaluations` | admin | Enqueue gateway evaluation |
| `POST /api/v1/benchmarks` | admin | Compatibility alias for evaluation enqueue |
| `GET /api/v1/evaluations/{run_id}` | read | Read evaluation results |
| `GET /api/v1/evaluations/{run_id}/report` | read | Download protected HTML report |
| `POST /api/v1/routers/train` | admin | Train a candidate from gateway evaluation |
| `POST /api/v1/routers/promote` | admin | Activate an eligible candidate |
| `POST /api/v1/routers/rollback` | admin | Reactivate a prior version |
| `GET /api/v1/routers/{kb}/versions` | read | List candidate/active/archive versions |
| `POST /api/v1/policies` | admin | Create an immutable policy definition |
| `GET /api/v1/policies` | read | List policy definitions |
| `GET /api/v1/policies/{id}` | read | Read one policy |
| `POST /api/v1/policies/{id}/promote` | admin | Make a policy usable by runs |
| `GET /api/v1/providers` | read | List provider availability/pricing metadata |
| `POST /api/v1/providers/test` | admin | Test provider connectivity |
| `POST /api/v1/api-keys` | admin | Create a scoped key; secret returned once |
| `GET /api/v1/api-keys` | admin | List metadata without secrets |
| `POST /api/v1/api-keys/{id}/rotate` | admin | Replace secret; new secret returned once |
| `DELETE /api/v1/api-keys/{id}` | admin | Disable a key |
| `GET /v1/models` | read | List `kb:<slug>:<policy>` models |
| `POST /v1/chat/completions` | write | OpenAI-compatible answer endpoint |

## Answers And Streaming

Use `stream_mode=none` for JSON, `verified` for lifecycle SSE followed by verified token chunks, or
`provisional` for live provider deltas. Provisional streaming must be enabled explicitly and clients
must handle a final retraction. Strict deployments keep it disabled.

Resume run events with:

```http
GET /api/v1/runs/{run_id}/events?after_sequence=17&follow=true
```

Every answer response includes decision status, citations, retrieval route, risk/evidence reports,
provider, policy snapshot, run ID, and cost. Read `status` before reading `answer`.

## Jobs And Idempotency

Upload, sync, evaluation, and router-training endpoints return `202` jobs. Send `Idempotency-Key`
when retrying a submission. Reusing a key with different payload is rejected; replaying the same
payload returns the existing job and does not enqueue it twice.

## Errors

Native errors use `{"code","message","details"}`. Common codes are `validation_error`,
`not_found`, `policy_rejected`, `budget_exceeded`, and `provider_unavailable`. Authentication and
scope errors use standard FastAPI `detail` responses with HTTP 401/403.
