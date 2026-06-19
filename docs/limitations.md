# Advantages And Limitations

## Advantages

- Admission status is enforced: abstained and blocked responses cannot leak answer text.
- Claims are tied to exact citation index, chunk, and source mappings before release.
- Query and retrieved-context risk checks short-circuit unsafe requests.
- Provider choice, budgets, context packing, usage, pricing, deadline, and circuit state are part of
  execution rather than reporting-only metadata.
- Runs have server-generated identities, ordered resumable events, traces, and cost records.
- Durable jobs use payload-bound idempotency, an outbox, cancellation, retries, and partial-success
  states.
- Router promotion uses gateway evaluation, confidence bounds, checksums, and rollback.
- The no-paid-provider demo is deterministic, multilingual, and starts with Docker Compose.
- Clean architecture keeps domain/application behavior independent of Qdrant, LiteLLM, MLflow,
  SQLAlchemy, Celery, and UI adapters.

## Disadvantages And Tradeoffs

- The default local semantic verifier is an embedding-plus-lexical baseline, not a formal proof or
  a full multilingual NLI system. Domain evaluation remains mandatory.
- Rule-based injection detection covers known patterns but is not a general prompt-injection
  security product.
- Verified streaming buffers answer text until verification; provisional streaming has lower
  latency but requires retraction-aware clients.
- Compose runs several stateful services and is heavier than a library-only RAG stack.
- First FastEmbed startup downloads models and consumes more memory than deterministic test mode.
- Cost accuracy depends on provider usage and configured, versioned prices.
- The included synthetic demo is useful for smoke/regression testing, not publishable superiority
  claims or a substitute for a held-out domain release set.
- Single-tenancy means one trust boundary; API-key scopes are not tenant RBAC.
- Uploaded text PDFs are supported, but OCR, scanned documents, tables, images, and office formats
  are deferred.
- PostgreSQL and Qdrant form a distributed corpus state; backup/recovery must reconcile versions.

## Production Posture

v0.2 is production-oriented for a hardened single-tenant deployment behind TLS. It is not an
unqualified production guarantee. Operators own provider/model selection, domain evaluation,
capacity planning, backups, incident response, data governance, and legal/model-license review.

Kubernetes, multi-tenant RBAC, OCR, a general security firewall, and managed cloud operations are
explicitly outside v0.2.
