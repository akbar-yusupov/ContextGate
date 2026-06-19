# Getting Started

## Requirements

- Docker Desktop or Docker Engine with Compose v2.
- At least 4 GB free memory for the deterministic local demo. FastEmbed deployments need more.
- Ports `8000`, `8001`, `5432`, `6379`, and `6333`, or overrides in an env file.

## Start The Demo

From the repository root, run one command:

```bash
docker compose --env-file .env.example --profile demo up --build
```

The profile starts PostgreSQL, authenticated Redis, Qdrant, the API, worker, Chainlit, and an
idempotent demo seeder. The default uses deterministic local embeddings and no paid provider.

Wait until `demo-init` prints `answered_status: answered` and `abstained_status: abstained`.

| Surface | Address |
|---|---|
| OpenAPI | http://localhost:8000/docs |
| Readiness | http://localhost:8000/ready |
| Chainlit | http://localhost:8001 |
| Qdrant | http://localhost:6333/dashboard |

The demo API key is `contextgate-dev-key`. It is not a production secret.

## Ask Two Questions

```bash
curl http://localhost:8000/api/v1/runs/answer \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"knowledge_base":"demo","query":"How can I cancel an order?","policy":"balanced"}'
```

The response should have `status: answered`, at least one citation, and a non-null
`evidence_report`.

```bash
curl http://localhost:8000/api/v1/runs/answer \
  -H "X-API-Key: contextgate-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"knowledge_base":"demo","query":"Do you accept cryptocurrency?","policy":"balanced"}'
```

The response should have `status: abstained`, an empty `answer`, and a stable
`abstention_reason`.

## Use Your Own Documents

Create a knowledge base through `/api/v1/knowledge-bases`, then upload PDF, Markdown, HTML, or TXT
to `/api/v1/knowledge-bases/{slug}/documents`. Poll the returned job at `/api/v1/jobs/{job_id}`.
The Chainlit UI can perform the same upload by attaching a file to a message.

Do not change vector dimensions to match the shape or size of your documents. Dimensions must match
the embedding models. If you switch embedding models, create a new knowledge base and ingest into
its new Qdrant collection. Chainlit shows structured worker failures when ingestion cannot complete.

## Stop Or Reset

```bash
docker compose --env-file .env.example --profile demo down
docker compose --env-file .env.example --profile demo down -v  # deletes demo data
```

Do not run the `-v` form against a deployment whose volumes have not been backed up.

For production configuration, continue with [Deployment](deployment.md). For failures, run
`ctxgate doctor` and use [Troubleshooting](troubleshooting.md).
