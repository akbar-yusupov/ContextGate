# Workflows

This page shows the common ContextGate workflows.

## Start The Stack

```bash
cp .env.example .env
docker compose up --build -d
curl http://127.0.0.1:8000/health
```

Default API key:

```text
contextgate-dev-key
```

MLflow starts with `CONTEXTGATE_MLFLOW_WORKERS=1` in local Compose to avoid slow multi-worker
startup. Increase it in `.env` only after the stack is stable and you need more concurrent MLflow
traffic.

## Run The Demo

```bash
docker compose exec api ctxgate ingest demo/documents --knowledge-base demo
```

This loads the included multilingual support knowledge base into the `demo` knowledge base.

Ask a grounded demo question:

```bash
curl http://localhost:8000/api/v1/runs/answer \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_base": "demo",
    "query": "How can I cancel an order?",
    "policy": "auto",
    "limit": 5
  }'
```

You should see `grounded: true`, `abstention_reason: null` and a citation to
`cancel-order-en.md`.

Ask an unanswerable demo question:

```bash
curl http://localhost:8000/api/v1/runs/answer \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_base": "demo",
    "query": "Do you accept cryptocurrency?",
    "policy": "auto",
    "limit": 5
  }'
```

You should see `provider: "abstention"`, `grounded: false`, no citations and a stable
`abstention_reason`.

For the full benchmark/router demo:

```bash
docker compose exec api ctxgate demo
```

It ingests multilingual support documents, prints one grounded answer and one abstention, runs the
QA Gate evaluation over `demo/benchmark.jsonl`, trains a router and keeps `balanced` as fallback if
release gates fail. The generated HTML report starts with false answers, false abstentions,
citation validity, latency and cost per answer.

## Create A Knowledge Base

```bash
curl http://localhost:8000/api/v1/knowledge-bases \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support",
    "slug": "support",
    "description": "Customer support knowledge base"
  }'
```

## Ingest Documents

CLI:

```bash
ctxgate ingest ./documents --knowledge-base support
```

API upload:

```bash
curl http://localhost:8000/api/v1/knowledge-bases/support/documents \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Idempotency-Key: support-docs-v1" \
  -F "file=@./documents/cancel-order.md"
```

Sync an existing Qdrant collection into a managed ContextGate collection:

```bash
curl http://localhost:8000/api/v1/knowledge-bases/support/sync-qdrant \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"source_collection": "existing_support_collection"}'
```

## Retrieve Without Generation

```bash
curl http://localhost:8000/api/v1/retrieve \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_base": "support",
    "query": "How can I cancel an order?",
    "policy": "balanced",
    "limit": 10
  }'
```

Use this when tuning retrieval before letting the gateway answer.

## Answer With Evidence Gate

```bash
curl http://localhost:8000/api/v1/runs/answer \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_base": "support",
    "query": "How can I cancel an order?",
    "policy": "auto",
    "latency_budget_ms": 800,
    "cost_budget_usd": 0.002,
    "debug": true
  }'
```

Read these fields first:

- `grounded`
- `evidence_score`
- `abstention_reason`
- `retrieval.trace_id`
- `selected_provider`
- `citations`

## Use OpenAI-Compatible Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kb:support:auto",
    "messages": [{"role": "user", "content": "How can I cancel an order?"}]
  }'
```

Model format:

```text
kb:<knowledge-base-slug>:<policy>
```

Examples:

- `kb:support:auto`
- `kb:support:fast`
- `kb:support:balanced`
- `kb:support:accurate`

## Inspect A Run

```bash
curl http://localhost:8000/api/v1/runs/{run_id}/trace \
  -H "X-API-Key: contextgate-dev-key"

curl http://localhost:8000/api/v1/runs/{run_id}/events \
  -H "X-API-Key: contextgate-dev-key"

curl http://localhost:8000/api/v1/runs/{run_id}/cost \
  -H "X-API-Key: contextgate-dev-key"
```

Use this to answer:

- why was this route chosen?
- what evidence was retrieved?
- why did the gateway abstain?
- which citations passed validation?
- what provider/cost was recorded?

## Evaluate The QA Gate

Dataset JSONL:

```json
{
  "id": "q-001",
  "group_id": "cancel-order",
  "query": "How can I cancel an order?",
  "language": "en",
  "relevant_chunk_ids": ["cancel-order-en:0"],
  "expected_facts": ["An order can be cancelled before it is handed to the courier."],
  "answerable": true,
  "tags": ["paraphrase", "cancel-order"]
}
```

Run the gateway-level benchmark:

```bash
ctxgate benchmark ./evaluation.jsonl --knowledge-base support --evaluate-answers
```

With `--evaluate-answers`, ContextGate calls the same `AnswerWithEvidence` runtime used by
`/api/v1/runs/answer` and `/v1/chat/completions`. The report answers:

- which answerable questions were abstained;
- which unanswerable questions were answered;
- which answers failed citation or fact-coverage checks;
- how retrieval policy choice affected answer/abstain behavior;
- what latency and estimated cost the gateway recorded.

Train and promote a router:

```bash
ctxgate router train BENCHMARK_RUN_ID --knowledge-base support
ctxgate router promote BENCHMARK_RUN_ID --knowledge-base support
```

Promotion should be treated as a release decision: quality, latency and cost gates must pass before
`auto` becomes better than a fixed fallback.
