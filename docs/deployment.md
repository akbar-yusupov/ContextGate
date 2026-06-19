# Deployment

ContextGate v0.2 is production-oriented for a hardened, single-tenant Docker Compose deployment.
It is not a managed service, multi-tenant authorization system, or unrestricted public edge API.

## Required Controls

1. Set `CONTEXTGATE_ENVIRONMENT=production`.
2. Replace the bootstrap API key, database password, and Redis password with secrets from a secret
   manager. Do not commit a production env file.
3. Terminate TLS at a trusted reverse proxy and restrict request/body/time limits there.
4. Do not publish PostgreSQL, Redis, Qdrant, or MLflow ports outside a private network.
5. Restrict `/metrics` to monitoring infrastructure.
6. Use `TRACE_CONTENT_MODE=metadata` when prompts or evidence contain sensitive data.
7. Back up PostgreSQL and all named application/vector/artifact volumes before migrations.
8. Configure model pricing before accepting hard cost budgets.
9. Run `ctxgate doctor` and the release checklist against the exact image digest.

Production startup rejects disabled authentication, the demo key, or a missing Redis password.
API keys should be least privilege and rotated through the authenticated key endpoints.

## Upgrade

```bash
docker compose pull
docker compose run --rm api \
  alembic -c src/contextgate/adapters/sqlalchemy/alembic.ini upgrade head
docker compose up -d
curl --fail https://contextgate.example/ready
```

The API command also runs migrations for simple single-host startup. Explicit migration is
recommended so a failed schema change is visible before traffic is switched.

## Capacity

The initial release target is one host, 100 concurrent clients, and a 50 request/second extractive
burst. Provider-backed latency depends on the provider and is measured separately. Tune PostgreSQL
connections, worker concurrency, reverse-proxy limits, and provider quotas from load results.

Use Kubernetes, multi-tenant isolation, OCR, and scanned-PDF pipelines only through downstream
extensions; they are outside v0.2 support.
