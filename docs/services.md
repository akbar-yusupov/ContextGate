# Services

The `demo` profile runs core services plus Chainlit and the one-shot seeder. MLflow is optional and
belongs to the `evaluation` profile.

| Service | Purpose | Depends on | Default port | Persistence | Failure effect |
|---|---|---|---:|---|---|
| `api` | Native/OpenAI APIs, admission graph, traces | PostgreSQL, Redis, Qdrant | 8000 | Database, data, reports | Requests stop; workers may finish queued work |
| `worker` | Ingestion, sync, benchmark, router tasks | API readiness, Redis, PostgreSQL, Qdrant | none | Shared data/reports | New jobs remain queued |
| `postgres` | Runs, events, costs, jobs, policies, keys | none | 5432 | `postgres-18-data` | API is not ready |
| `redis` | Rate limits, Celery broker/results | none | 6379 | `redis-data` | API fails closed in Compose; jobs stop |
| `qdrant` | Dense, sparse, and late-interaction vectors | none | 6333 | `qdrant-data` | Retrieval and readiness fail |
| `chainlit` | Optional operator console | API | 8001 | none | APIs remain available |
| `demo-init` | Seeds and verifies the included demo once | API | none | Shared application volumes | Demo is not considered ready |
| `mlflow` | Optional evaluation tracking/artifacts | PostgreSQL | 5000 | `mlflow-artifacts` | Answer traffic remains available; health fails without its database |

## Profiles

- No profile: core API dependencies and worker.
- `demo`: core services, Chainlit, and `demo-init`.
- `ui`: Chainlit without demo seeding.
- `evaluation`: MLflow.

## Health And Logs

- `/health` proves the API process is alive.
- `/ready` checks PostgreSQL, Redis, and Qdrant.
- MLflow health checks its HTTP endpoint and PostgreSQL tracking store.
- Compose waits for API readiness before the worker, UI, or seeder starts.
- Inspect a service with `docker compose logs -f api` or replace `api` with its service name.

Application services run as a non-root user with all Linux capabilities dropped and
`no-new-privileges`. PostgreSQL, Redis, and Qdrant are bound to `127.0.0.1` by default; production
deployments should not publish these ports publicly.
