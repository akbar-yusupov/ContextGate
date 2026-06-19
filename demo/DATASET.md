# ContextGate multilingual demo dataset

This synthetic-but-curated dataset exercises retrieval behavior rather than claiming to represent
real production traffic.

Use this dataset when starting ContextGate for the first time. It gives you:

- known answerable questions that should return grounded answers with citations;
- known unanswerable questions that should return structured abstentions;
- a small benchmark for retrieval/router smoke tests.

Quick start:

```bash
docker compose --env-file .env.example --profile demo up --build
```

Then ask:

- grounded: `How can I cancel an order?`
- unanswerable: `Do you accept cryptocurrency?`

See [demo/README.md](README.md) for copy-paste API calls.

## Composition

- 150 queries: 50 English, 50 Russian, and 50 Uzbek.
- 27 short policy documents: nine topics per language.
- 15 explicitly unanswerable queries.
- Query tags cover exact phrasing, paraphrase, morphology, multi-intent, cross-language terms,
  and abstention.

Every answerable qrel maps to a deterministic chunk ID (`<document-stem>:0`), and every expected
fact appears verbatim in its source document. CI verifies those invariants.

## Intended use

- Test the evidence gate with one grounded answer and one abstention.
- Smoke-test dense, sparse, hybrid, late-interaction and routing code.
- Compare regressions across model or policy changes.
- Demonstrate language-specific failure analysis.

## Limitations

- The documents and questions are authored examples, not sampled customer data.
- It is too small for publishable model comparisons.
- Uzbek morphology and code-switching coverage is intentionally basic.
- The default Apache-2.0 ColBERT model is English-only; RU/UZ late interaction requires an
  explicitly configured compatible model.

Any benchmark published from this dataset must include hardware, model versions, all languages,
and failed queries. Production claims require a domain-specific held-out dataset.
