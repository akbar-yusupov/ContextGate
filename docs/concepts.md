# Concepts

This page explains the terms exposed by ContextGate APIs, traces and reports.

## Evidence Gate

The evidence gate decides whether generation is allowed and whether a generated draft may be
released.

It uses:

- `answerability_score`: does retrieved context appear capable of answering the query?
- `coverage_score`: how much of the query is covered by retrieved context?
- `support_score`: how much of the draft answer is supported by retrieved context?
- `evidence_score`: combined score used by the runtime gate.

After generation, every factual claim must resolve to cited evidence and pass the configured local
semantic verifier. If retrieval is empty, evidence is weak, risk policy blocks the input, or claim
verification fails, ContextGate releases no answer text.

Final statuses are `answered`, `abstained`, and `blocked`. `grounded` remains a compatibility
projection and is true only for `answered` responses.

## Abstention

An abstention is a successful gateway decision, not an internal error. It means ContextGate chose not
to answer because the evidence policy did not allow generation.

Stable reasons:

- `retrieval_empty`: no usable retrieved evidence.
- `low_coverage`: retrieved evidence does not cover the query enough.
- `low_support`: retrieved evidence does not support the draft answer enough.
- `invalid_citations`: generated citations do not map to retrieved chunks/ranks.
- `budget_exceeded`: policy budget prevents generation.
- `latency_budget_exceeded`: the end-to-end deadline expired.
- `unsupported_claim`: at least one claim was not supported by its cited evidence.
- `contradiction`: cited evidence contradicts a claim.
- `unsafe_query` / `unsafe_context`: risk policy blocked untrusted instructions.
- `provider_unavailable` / `verification_unavailable`: a required admission dependency was absent.

Applications can branch on these values directly.

## Citations

A citation is valid only when its index resolves to the same retrieved chunk and source. Each claim
is then checked against only the chunks it cites.

If a generated answer cites a missing chunk or impossible rank, the answer is marked `grounded=false`
and the abstention reason becomes `invalid_citations`.

## Claim Verification

The no-provider baseline performs deterministic exact citation mapping and lexical claim checks.
The default composed service additionally uses the configured multilingual embedding model for
local semantic support scoring. The `ClaimVerifier` port can be replaced by an NLI or structured
judge adapter without changing admission behavior. Verification fails closed.

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
risk_checked
retrieval_hit
evidence_scored
provider_selected
token_delta
citation_verified
final
```

Traces are meant for debugging, evaluation and regression review.

`CONTEXTGATE_TRACE_CONTENT_MODE=metadata` stores a query hash rather than raw query text. Run,
event, and cost records are purged after `CONTEXTGATE_TRACE_RETENTION_DAYS`.
