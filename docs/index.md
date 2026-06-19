# ContextGate Documentation

ContextGate is a production-oriented, single-tenant RAG admission gateway. Start with the demo,
then use the references that match your role.

## First Steps

- [Getting started](getting-started.md): one-command Docker demo and first API calls.
- [Workflows](workflows.md): ingest, retrieve, answer, inspect, evaluate, and promote.
- [Troubleshooting](troubleshooting.md): startup, provider, ingestion, and model-download failures.

## Product And Design

- [Product](product.md): intended users, core value, and non-goals.
- [Concepts](concepts.md): admission decisions, evidence, citations, policies, and traces.
- [Architecture](architecture.md): boundaries, runtime graph, adapters, jobs, and failure behavior.
- [Advantages and limitations](limitations.md): strengths, costs, risks, and deferred scope.

## References

- [Services](services.md): every Compose service, dependency, port, and failure effect.
- [Entities](entities.md): persisted and runtime domain objects.
- [API](api.md): endpoint groups, scopes, streaming, idempotency, and errors.
- [Configuration](configuration.md): environment variables and production recommendations.
- [Repository map](repository-map.md): purpose of each maintained directory and file group.
- [AI integration](ai-integration.md): native and OpenAI-compatible developer examples.

## Operating And Releasing

- [Deployment](deployment.md): hardened single-host deployment prerequisites.
- [Operations](operations.md): readiness, logs, metrics, backup, recovery, and rollback.
- [Security policy](../SECURITY.md): vulnerability reporting and deployment controls.
- [Migration from v0.1](migration-v0.2.md): breaking alpha contract changes.
- [Launch notes](launch.md): fact-checked public description and LinkedIn draft.

The interactive OpenAPI reference is available at `http://localhost:8000/docs` when the API is
running.
