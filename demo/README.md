# Demo Data

The demo data is a small multilingual support knowledge base for first-time ContextGate runs.

It is designed to show the core product behavior:

```text
ingest documents -> retrieve evidence -> answer with citations or abstain -> inspect QA report
```

## Files

- `demo/documents/`: 27 Markdown policy documents.
- `demo/benchmark.jsonl`: 150 labeled evaluation queries.
- `demo/DATASET.md`: dataset composition, limitations and benchmark notes.

Languages:

- English: 9 documents, 50 queries.
- Russian: 9 documents, 50 queries.
- Uzbek: 9 documents, 50 queries.

The topics are order cancellation, returns, delivery, payment, password reset, subscription,
invoices, support hours and data deletion.

## First Run With Docker

From the repository root:

```bash
cp .env.example .env
docker compose up --build -d postgres redis qdrant mlflow api
docker compose exec api ctxgate ingest demo/documents --knowledge-base demo
```

The first ingest downloads embedding models into the Docker `model-cache` volume. Later runs are
faster.

Ask a grounded question:

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

Expected signal:

- `grounded: true`
- `provider: "extractive"` unless an LLM provider is configured
- at least one citation pointing to `cancel-order-en.md`
- `abstention_reason: null`

Ask an unanswerable question:

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

Expected signal:

- `provider: "abstention"`
- `grounded: false`
- no citations
- stable `abstention_reason`, usually `low_coverage` or `retrieval_empty`

## Full Demo Command

```bash
docker compose exec api ctxgate demo
```

This command ingests `demo/documents`, prints one grounded answer and one abstention, runs the
gateway-level QA evaluation over `demo/benchmark.jsonl`, trains the retrieval router and keeps
`balanced` as fallback if promotion gates fail.

The report path printed by the command points to an HTML report with:

- answer rate and abstention rate;
- false answer and false abstention rates;
- citation validity and grounded answer rate;
- failed cases with query, policy, reason and evidence score;
- retrieval metrics lower down as supporting detail.

## Local Python Smoke Mode

Use deterministic embeddings when you want a quick no-download smoke test.

Docker:

```bash
# edit .env
CONTEXTGATE_EMBEDDING_BACKEND=deterministic
CONTEXTGATE_DENSE_DIMENSION=64
CONTEXTGATE_LATE_DIMENSION=32
```

Then restart the stack and ingest again.

Local Python:

```bash
CONTEXTGATE_EMBEDDING_BACKEND=deterministic \
CONTEXTGATE_DENSE_DIMENSION=64 \
CONTEXTGATE_LATE_DIMENSION=32 \
uv run ctxgate demo
```

Deterministic mode is useful for development checks. For realistic retrieval behavior, use the
default FastEmbed models.
