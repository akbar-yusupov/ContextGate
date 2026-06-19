# Entities

## Knowledge And Ingestion

| Entity | Meaning | Lifecycle |
|---|---|---|
| Knowledge base | Named retrieval boundary and Qdrant collection | Created once; corpus version advances on ingestion |
| Document | One content-hash and pipeline-version record | `processing` to `ready` or `failed` |
| Chunk | Retrieval unit with language, source, metadata, and stable ID | Rebuilt when document/pipeline version changes |
| Job | Durable async operation | `queued`, `running`, `succeeded`, `succeeded_with_errors`, `failed`, or `cancelled` |
| Job outbox | Transactional intent to dispatch one job | `pending` to `dispatched`; replayed after restart |

## Answer Runtime

| Entity | Meaning |
|---|---|
| Run | Server-generated execution identity; client `X-Request-ID` is correlation only |
| Retrieval result | Ranked hits, selected policy, route reason, scores, timings, and corpus version |
| Risk report | Query/context risk score, matched rules, and blocking decision |
| Evidence report | Per-claim citations, resolved chunks, entailment/contradiction scores, verifier version, and repair result |
| Citation | Exact `(index, chunk_id, source)` mapping to a retrieved hit |
| Run event | Ordered lifecycle record, resumable by sequence number |
| Cost record | Provider/model usage and recorded estimated/actual USD |

The primary decision is `answered`, `abstained`, or `blocked`. Abstained and blocked runs always
return an empty answer. Operational failures use typed HTTP errors instead of admission statuses.

## Configuration And Evaluation

| Entity | Meaning |
|---|---|
| Gateway policy | Immutable retrieval, provider, latency, and cost snapshot applied to a run |
| Router version | Checksummed learned routing artifact with candidate/active/archived state |
| API key | Hashed credential with `read`, `write`, and/or `admin` scopes; plaintext is returned once |
| Evaluation | MLflow run plus `results.json` and protected HTML report |
| Evaluation case | Labeled answerable/unanswerable query with language, expected facts, tags, and qrels |

PostgreSQL is authoritative for active policy/router state. Qdrant is authoritative for vector
payloads. Traces record the policy, corpus, provider/model, pricing, router, and verifier versions
used for reproducibility.
