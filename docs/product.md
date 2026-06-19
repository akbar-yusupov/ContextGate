# Product

ContextGate is a **RAG QA Gate** for RAG engineers.

Its core promise is simple:

> Return grounded answers only. If the evidence is not strong enough, abstain with a reason.

The gateway sits between your application, your knowledge base and your LLM provider. For each
question it retrieves context, scores whether that context is good enough, validates citations and
claims, records trace/cost metadata, then returns either a grounded answer or a structured
abstention.

## Why This Is Not Just A Prompt

Claude, OpenAI and other LLMs can answer one request when you paste in context. That is not the hard
production problem ContextGate targets.

The hard problem for RAG teams is operational QA:

- Should this query be answered at all?
- Can the system prove which evidence allowed the answer?
- Did the model cite retrieved chunks, or invent citation-looking text?
- Which policy/provider was selected under latency and cost budgets?
- Did a retrieval/router change increase false answers or false abstentions?
- Can the team inspect failures before promoting a new policy?

ContextGate turns those checks into a repeatable API and evaluation loop.

## Who It Is For

ContextGate is for teams that already understand RAG basics and need the operational layer around
it:

- RAG engineers comparing retrieval policies before serving answers.
- AI engineers who need evidence scores, citations and trace inspection.
- Backend engineers turning RAG into an API with jobs, rate limits, keys and metrics.
- Teams that want OpenAI-compatible chat over their documents without hiding grounding metadata.

## Problem

Many RAG systems fail in production for reasons that are not visible in a simple chat demo:

- retrieval returns plausible but insufficient context;
- generated answers cite chunks that were not retrieved;
- unanswerable questions still get confident responses;
- teams cannot tell whether a new policy increased false answers or false abstentions;
- teams cannot compare quality, latency and cost across retrieval/provider policies;
- support engineers cannot inspect why a route or answer was chosen.

ContextGate treats evidence as a runtime gate, not as an afterthought.

## Product Loop

```text
Connect documents
-> Ask a question
-> Retrieve evidence
-> Gate generation
-> Answer with citations or abstain
-> Inspect trace, cost and failure reason
-> Evaluate and promote policy
```

## What It Does

- Ingests PDF, Markdown, HTML and TXT.
- Stores dense, sparse and late-interaction vectors in Qdrant.
- Runs fixed retrieval policies and an `auto` router.
- Uses LangGraph for the answer runtime.
- Validates evidence, citations and unsupported claims.
- Serves native APIs and OpenAI-compatible chat completions.
- Logs traces, cost records, QA Gate reports and MLflow artifacts.

## QA Gate Report

When `ctxgate benchmark ... --evaluate-answers` runs, ContextGate evaluates the canonical answer
runtime instead of a separate benchmark-only generation path. The report starts with gateway metrics:

- answer rate and abstention rate;
- correct abstention rate for unanswerable queries;
- false answer rate for unanswerable queries that were answered as grounded;
- false abstention rate for answerable queries that were blocked;
- grounded answer rate and citation validity;
- latency and estimated cost per answer;
- failed cases with query, policy, reason, evidence score and expected facts.

Retrieval metrics remain in the report, but they support the main question: did the QA gate make the
right answer/abstain decision?

## What It Does Not Do

- It is not a chatbot UI framework.
- It is not a CMS.
- It does not do OCR or scanned document extraction in v0.2.
- It does not provide multi-tenant RBAC in v0.2.
- It is not a full prompt-injection defense platform.
- It does not claim benchmark superiority from the included demo dataset.

## Why RAG Engineers Should Care

ContextGate is useful when the hard part is no longer "can I call an LLM?" but:

- "Should I call the LLM for this query?"
- "Can I explain why this answer was allowed?"
- "Which retrieval policy gives enough quality within latency budget?"
- "What exact evidence supported this response?"
- "How do I detect regressions before promoting a new router?"

Those questions are the product surface.

## Roadmap

- `v0.1`: initial QA gate, citations, traces and demo evaluation.
- `v0.2`: enforced admission states, claim evidence reports, real provider/cost routing, durable
  outbox jobs, scoped keys, resumable events and confidence-bound router promotion.
- `v0.3`: stronger claim verification, adversarial cases and security-focused hooks.
