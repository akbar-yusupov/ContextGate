# ContextGate v0.2 Launch Notes

## Public Description

ContextGate is an open-source RAG admission gateway. It decides whether retrieved evidence is
strong enough to answer, validates claim-to-citation mappings, enforces answer/abstain/block
decisions, and records the trace, policy, provider usage, and cost needed to audit the result.

It is production-oriented for hardened single-tenant Docker Compose deployments. It does not claim
to be a managed service, multi-tenant platform, formal proof system, or complete injection defense.

## LinkedIn Draft

I am publishing ContextGate v0.2, an open-source evidence-gated RAG gateway.

The project focuses on a question that normal RAG demos usually skip: should the system be allowed
to answer at all?

ContextGate retrieves evidence, applies query and context risk gates, selects a provider under
latency/cost constraints, validates every citation against the retrieved chunk, checks claim
support, and returns one explicit status: answered, abstained, or blocked. Every run has resumable
events, trace and cost records, and a versioned policy/evidence report.

The Docker demo works without a paid model and includes PostgreSQL, Redis, Qdrant, a Celery worker,
FastAPI/OpenAI-compatible endpoints, and an optional Chainlit operator UI. MLflow is optional for
evaluation and router lifecycle work.

This is a production-oriented single-tenant release, not an unrestricted production guarantee.
The documentation states the verifier, security, deployment, and dataset limitations directly.

Demo: `docker compose --env-file .env.example --profile demo up --build`

Add the repository URL, release-report link, exact test count, evaluation confidence bounds, image
digest, and measured load results immediately before publishing. Do not insert estimates.

## Publication Checklist

- Link to README, architecture, getting started, limitations, security, and release report.
- State hardware, model versions, dataset composition, and date for every metric.
- Do not describe the 150-query synthetic demo as a production benchmark.
- Do not use "production ready" without the "production-oriented single-tenant" qualification.
