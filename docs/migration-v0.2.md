# Migrating from v0.1 to v0.2

ContextGate v0.2 intentionally breaks alpha contracts that could misrepresent grounding, cost, or
run identity.

The `0008_cost_request_id_length` revision expands cost-ledger correlation IDs from 64 to 128
characters. This fixes benchmark request IDs such as `eval-<run>-<case>-<policy>` and aligns the
ledger with gateway runs. Inputs longer than 128 characters receive a deterministic hash suffix.

1. Back up PostgreSQL, Qdrant, `data/routers`, reports, and MLflow artifacts.
2. Upgrade dependencies and run `alembic -c src/contextgate/adapters/sqlalchemy/alembic.ini upgrade head`.
3. Configure non-default production secrets. Production startup now rejects disabled auth, the
   development API key, and a missing Redis password.
4. Configure model input/output prices before sending a hard `cost_budget_usd`. Unknown-price
   models are rejected under hard budgets.
5. Replace `POST /api/v1/answer` with `POST /api/v1/runs/answer`. Client request IDs are correlation
   values; the server always creates the run ID.
6. Read `status` as the primary decision. `abstained` and `blocked` responses have an empty `answer`;
   inspect `abstention_reason`, `evidence_report`, and `risk_report`.
7. Upload benchmark JSONL through `POST /api/v1/evaluations/datasets` and submit the returned
   `dataset_id`; arbitrary server-local evaluation paths are no longer accepted over HTTP.
8. Use `stream_mode=verified` for strict admission. Enable provisional streaming only when clients
   understand provisional deltas and retraction events.
9. Re-run gateway answer evaluation before router training. Retrieval-only and undersized datasets
   can train candidates but cannot be promoted.
10. Run `ctxgate doctor`, then execute the full test and benchmark gates before serving traffic.

The migration adds foreign keys. Existing orphan events, costs, documents, or router rows must be
removed or repaired before applying the migration.
