# AI And LLM Integration

## OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="contextgate-dev-key")
response = client.chat.completions.create(
    model="kb:demo:balanced",
    messages=[{"role": "user", "content": "How can I cancel an order?"}],
)
print(response.choices[0].message.content)
print(response.model_extra["contextgate"])
```

JavaScript uses the same model and base URL with the official `openai` package. ContextGate accepts
system/developer messages as lower-priority request instructions; they cannot override grounding
rules.

## Native Contract

Use the native endpoint when the application needs filters, explicit budgets, evidence/risk
reports, or policy IDs. Always branch on `status` before consuming `answer`:

```python
import httpx

result = httpx.post(
    "http://localhost:8000/api/v1/runs/answer",
    headers={"X-API-Key": "contextgate-dev-key"},
    json={"knowledge_base": "demo", "query": "How can I cancel?", "policy": "auto"},
).raise_for_status().json()

if result["status"] == "answered":
    use_answer(result["answer"], result["citations"])
else:
    explain_no_answer(result["status"], result["abstention_reason"])
```

Do not treat an empty answer as a transport error. Do not retry `blocked` or evidence-based
abstentions automatically. Retry typed dependency failures with bounded exponential backoff and
the same correlation ID; use idempotency keys for job submissions.

## Streaming

- `verified`: live lifecycle events, then verified answer chunks. This is the default for agents
  that may take actions from generated text.
- `provisional`: true provider deltas marked provisional, followed by final decision or retraction.
  Enable only when the client can discard provisional text.
- `/runs/{id}/events`: resume with `after_sequence` after disconnect.

The OpenAI final chunk carries a `contextgate` block with status, evidence, citations, selected
policy/provider, trace/run IDs, and cost. Agent frameworks must retain that metadata for audit.

## Repository Coding Agents

`AGENTS.md` is the canonical implementation guide. `CLAUDE.md`, `GEMINI.md`, Copilot, Cursor, and
`llms.txt` files point to the same invariants. Agents must preserve `answered | abstained | blocked`,
fail verification closed, keep adapters behind ports, add tests at the affected boundary, and never
use demo quality numbers as production claims.
