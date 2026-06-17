# Concepts

This page explains the terms exposed by ContextGate APIs, traces and reports.

## Evidence Gate

The evidence gate decides whether generation is allowed.

It uses:

- `answerability_score`: does retrieved context appear capable of answering the query?
- `coverage_score`: how much of the query is covered by retrieved context?
- `support_score`: how much of the draft answer is supported by retrieved context?
- `evidence_score`: combined score used by the runtime gate.

If retrieval is empty, evidence is weak, or citations fail validation, ContextGate abstains.

## Abstention

An abstention is a successful gateway decision, not an internal error. It means ContextGate chose not
to answer because the evidence policy did not allow generation.

Stable reasons:

- `retrieval_empty`: no usable retrieved evidence.
- `low_coverage`: retrieved evidence does not cover the query enough.
- `low_support`: retrieved evidence does not support the draft answer enough.
- `invalid_citations`: generated citations do not map to retrieved chunks/ranks.
- `budget_exceeded`: policy budget prevents generation.

Applications can branch on these values directly.

## Citations

A citation is valid only if it references a retrieved chunk and a valid citation index. ContextGate
does not treat citation-looking text as trusted by default.

If a generated answer cites a missing chunk or impossible rank, the answer is marked `grounded=false`
and the abstention reason becomes `invalid_citations`.

## Unsupported Claims

The v0.1 unsupported-claim detector is lexical and intentionally simple. It surfaces terms in the
answer that are absent from retrieved context. The detector is isolated in the domain layer so it can
be replaced by a stronger verifier later.

Use it as a failure-analysis signal, not as a final safety classifier.

## Retrieval Policies

- `fast`: dense retrieval for low latency.
- `balanced`: dense + sparse/BM25, reciprocal rank fusion and late-interaction reranking when
  supported.
- `accurate`: larger prefetch and more expensive reranking path.
- `auto`: router-selected policy with `balanced` fallback when no promoted router exists.

The answer runtime always records the requested policy, selected policy and route reason.

## Provider Routing

ContextGate supports:

- `extractive`: no paid LLM; answer is assembled from retrieved evidence.
- LiteLLM/OpenAI-compatible models when configured.
- local/Ollama-style provider hooks.

If evidence is insufficient, the runtime abstains and records `selected_provider=abstention`.

## Traces

Each answer run can be inspected through:

```http
GET /api/v1/runs/{run_id}/trace
GET /api/v1/runs/{run_id}/events
GET /api/v1/runs/{run_id}/cost
```

Typical event order:

```text
query_analyzed
retrieval_started
retrieval_hit
evidence_scored
provider_selected
token_delta
citation_verified
final
```

Traces are meant for debugging, evaluation and regression review.
