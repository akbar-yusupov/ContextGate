# Security policy

ContextGate `v0.2` is a production-oriented single-tenant gateway. Put it behind a TLS reverse
proxy and do not expose stateful dependencies directly to the public internet.

## Reporting

Use GitHub private vulnerability reporting. Include reproduction steps, affected versions, and
potential impact. Do not open public issues for unpatched vulnerabilities.

## Deployment requirements

- Replace the development API key and terminate TLS at a trusted proxy.
- Set `CONTEXTGATE_ENVIRONMENT=production`; startup rejects disabled authentication, the default
  development key, or a missing Redis password in production mode.
- Use scoped keys and rotate them through the authenticated `/api/v1/api-keys` endpoints. Plaintext
  key material is returned only on creation or rotation.
- Restrict Qdrant, PostgreSQL, Redis, and MLflow to a private network.
- Treat uploaded documents, prompts, traces, and model artifacts as sensitive data.
- Pin and scan container images before deployment.
- Use a dedicated database and object store for MLflow in shared environments.
- Review every configured model license.
- Keep provisional streaming disabled unless clients explicitly handle retraction events.
- Restrict `/metrics` and evaluation reports to trusted operators.
- Run the API/worker images as the bundled non-root user with dropped capabilities and
  `no-new-privileges`.
- Use `CONTEXTGATE_TRACE_CONTENT_MODE=metadata` for sensitive workloads and set an appropriate trace
  retention period.

The existing-Qdrant adapter copies data and does not mutate the source collection. It does not
provide tenant isolation or sanitize source payloads.
