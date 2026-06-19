# Troubleshooting

Start with:

```bash
ctxgate doctor
docker compose ps
docker compose logs --tail=200 api worker demo-init
```

## Common Failures

| Symptom | Cause | Resolution |
|---|---|---|
| Compose says a required variable is missing | No env file | Use `--env-file .env.example` for demo or create a private production env file |
| `/ready` returns 503 | PostgreSQL, Redis, or Qdrant unavailable | Inspect the named dependency and verify password/service/port settings |
| Redis authentication error | Client/server passwords differ | Recreate demo volumes or make both `REDIS_PASSWORD` values match |
| `demo-init` exits nonzero | Ingestion or answer/abstention acceptance failed | Read its JSON result and worker/API logs |
| First start appears slow | FastEmbed model download | Watch API/worker logs and preserve `model-cache` |
| MLflow logs `failed to resolve host 'postgres'` | MLflow/PostgreSQL are not in the same Compose project/network, PostgreSQL stopped, or stale logs are being viewed | Start both with the same project and env file; confirm the MLflow container creation time with `docker compose ps -a` |
| Upload reports an incompatible Qdrant dimension | The collection was created by another embedding model/dimension contract | Restore the original model settings or create a new knowledge base; reset volumes only for disposable demo data |
| FastEmbed reports configured/emitted dimension mismatch | A dimension was edited without selecting a model that emits that dimension | Pair `DENSE_MODEL` with its actual `DENSE_DIMENSION` and `LATE_MODEL` with its actual `LATE_DIMENSION`; documents do not determine vector size |
| Ingestion reports `value too long` | A pipeline/idempotency/config identifier exceeds its schema limit | Keep `PIPELINE_VERSION` at 32 characters or fewer and idempotency keys at 128 or fewer; document-derived IDs are bounded automatically |
| Upload returns 422 | Size, suffix, PDF signature, encryption, page, binary, or UTF-8 validation | Convert to supported PDF/Markdown/HTML/TXT within configured limits |
| Job remains queued | Worker not running or Redis unavailable | Start worker, check broker, and inspect outbox replay logs |
| Hard budget always abstains | Model pricing unknown or projected cost too high | Configure input/output rates or remove the hard budget |
| `provisional` returns 409 | Strict streaming policy | Use `verified` or explicitly enable provisional/retraction handling |
| Answer is empty with HTTP 200 | Admission status is abstained/blocked | Read `status`, `abstention_reason`, `risk_report`, and `evidence_report` |
| Filter rejected | Qdrant strict mode lacks payload index | Add field/type to `INDEXED_METADATA_FIELDS` and reindex |
| Evaluation report is 404 | Job ID used instead of evaluation run ID | Poll the job and use `job.result.run_id` |
| Router will not promote | Release-set size/confidence/language/adversarial gate failed | Inspect `promotion_failures` in evaluation/training output |

To reset only the disposable demo, run
`docker compose --env-file .env.example --profile '*' down -v --remove-orphans`. Enabling all
profiles is important: plain `docker compose down -v` does not remove containers belonging to an
inactive profile, so an old MLflow process can reconnect to a newly empty PostgreSQL volume. This
deletes all project volumes
and must not be used for production recovery.

MLflow is optional and is not started by the `demo` profile. Start it together with its database:

```bash
docker compose --env-file .env.example --profile evaluation up -d postgres mlflow
docker compose --env-file .env.example --profile evaluation ps
```

Use the same `--env-file` for every command in one Compose project. If you created a customized
`.env`, replace `.env.example` with `.env` in both commands. Mixing them can change database/Redis
ports, recreate dependencies, and make historical container logs look like a current DNS failure.

Its health check verifies both HTTP and the tracking database, so a healthy container has a usable
PostgreSQL connection.

## MLflow model alias reports an integer/varchar error

ContextGate uses the MLflow HTTP service for tracking and artifacts, but connects to the configured
`CONTEXTGATE_MLFLOW_REGISTRY_STORE_URI` for model aliases and version tags. This avoids a MLflow
3.13 REST serialization mismatch with PostgreSQL model-version integers. Leave the setting empty in
Compose to use the generated MLflow PostgreSQL URI, or provide a reachable SQL registry URI. After
rebuilding, recreate both `api` and `mlflow` so their locked MLflow versions match.
