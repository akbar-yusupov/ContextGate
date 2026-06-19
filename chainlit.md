# ContextGate

Operator console for the ContextGate v0.2 RAG admission gateway.

Use the settings panel to choose:

- **Chat** for grounded answers with citations.
- **Retrieval Inspector** for ranked chunks, scores, and route timings.
- **Policy Compare** to compare fast, balanced, and accurate policies.

The `Why this route?` panel exposes the selected policy, SLO budget, predicted quality, node
latencies, and MLflow trace ID.

Attach a supported document to ingest it into the selected knowledge base. Use
`/kb create <slug> <display name>` to create a knowledge base, then reopen settings to select it.

