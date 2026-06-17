# Security policy

ContextGate `v0.1` is single-tenant and should not be exposed directly to the public internet.

## Reporting

Use GitHub private vulnerability reporting. Include reproduction steps, affected versions, and
potential impact. Do not open public issues for unpatched vulnerabilities.

## Deployment requirements

- Replace the development API key and terminate TLS at a trusted proxy.
- Restrict Qdrant, PostgreSQL, Redis, and MLflow to a private network.
- Treat uploaded documents, prompts, traces, and model artifacts as sensitive data.
- Pin and scan container images before deployment.
- Use a dedicated database and object store for MLflow in shared environments.
- Review every configured model license.

The existing-Qdrant adapter copies data and does not mutate the source collection. It does not
provide tenant isolation or sanitize source payloads.
