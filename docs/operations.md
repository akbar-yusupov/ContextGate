# Operations

## Health, Logs, And Metrics

- Alert on `/ready` failures; `/health` alone is insufficient.
- Application HTTP logs are JSON and include request ID, normalized route, status, and duration.
- Pass `X-Request-ID` for correlation; the server still generates a separate run ID.
- Scrape `/metrics` for HTTP latency/status, admission decisions, retrieval routes, provider
  latency/failures/circuit openings, recorded costs, and ingestion volume.
- Watch Celery queue depth, job age, PostgreSQL connections, Redis memory, Qdrant storage, disk
  space, and container restarts at the platform layer.

## Backup And Restore

Back up PostgreSQL, `qdrant-data`, `contextgate-data`, `contextgate-reports`, `model-cache`, and
MLflow artifacts when evaluation is enabled. Redis contains durable broker state but PostgreSQL's
job/outbox rows are the recovery source for undispatched work.

Restore PostgreSQL and Qdrant from a consistent checkpoint. Run migrations, start dependencies,
start the API, confirm `/ready`, then start workers. Reconcile corpus versions before serving
answers if either store was restored independently.

## Retention And Redaction

Set `TRACE_RETENTION_DAYS`; startup purges old trace records. `TRACE_CONTENT_MODE=metadata` stores
content hashes instead of query/retrieval text in traces. Application logs must not include API
keys, document bodies, prompts, or provider secrets.

## Incidents

- Provider unavailable: circuit breaker opens and requests abstain with `provider_unavailable`.
- Verifier unavailable: fail closed with `verification_unavailable`.
- Redis unavailable: Compose fails rate limiting closed and workers stop consuming.
- Qdrant unavailable: readiness fails and retrieval is unavailable.
- Worker loss: late acknowledgements return eligible tasks to the broker; idempotency prevents
  duplicate submissions.
- Bad router: call rollback with the prior run ID and confirm PostgreSQL active status/checksum.

## Release Evidence

Run unit/static checks, clean Compose E2E, release evaluation, security tests, image scan, SBOM, and
`scripts/load_smoke.py`. The release SLO profile sends 500 requests at 50 RPS through a 100-user
pool using four API workers, deterministic embeddings, the `fast` retrieval policy, and at least
six host CPUs. Its limits are 2s p95, 4s p99, and zero failures. The separate 100-request
simultaneous overload probe uses 3s p95 and 4s p99 limits. Store the generated release report with
the image digest and hardware description; never publish placeholder latency or quality numbers.

Retrieval workers cache knowledge-base metadata for one second. A corpus-version update can take
up to one second to appear in a new trace, while retrieval continues against the same collection.
Bootstrap-key authentication is stateless; managed API keys remain immediately revocable through
database lookup.
