from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

import contextgate.apps.api.main as api_module
from contextgate.apps.api.main import app
from contextgate.apps.api.schemas import (
    AnswerResponse,
    Citation,
    RetrievalHit,
    RetrieveResponse,
    RouteDecision,
)


def test_openapi_groups_endpoints_for_new_users() -> None:
    schema = app.openapi()
    tag_names = [tag["name"] for tag in schema["tags"]]

    assert tag_names == [
        "System",
        "Knowledge Bases",
        "Ingestion",
        "Jobs",
        "Retrieval",
        "Answer Runtime",
        "Runs/Traces",
        "Evaluations",
        "Routers",
        "Policies",
        "Providers",
        "API Keys",
        "OpenAI Compatibility",
    ]
    assert schema["paths"]["/api/v1/runs/answer"]["post"]["tags"] == ["Answer Runtime"]
    assert "/api/v1/answer" not in schema["paths"]
    assert schema["paths"]["/v1/chat/completions"]["post"]["tags"] == ["OpenAI Compatibility"]
    operation_ids = [
        operation["operationId"]
        for path in schema["paths"].values()
        for operation in path.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids))


def test_resource_discovery_contracts() -> None:
    slug = f"discover-{uuid4().hex[:10]}"
    policy_name = f"policy-{uuid4().hex[:10]}"
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "Discover", "slug": slug, "description": "Discovery API"},
        )
        policy = client.post(
            "/api/v1/policies",
            json={
                "name": policy_name,
                "retrieval_policy": "balanced",
                "provider_policy": "extractive",
                "latency_budget_ms": 1000,
            },
        )
        knowledge_bases = client.get("/api/v1/knowledge-bases")
        knowledge_base = client.get(f"/api/v1/knowledge-bases/{slug}")
        documents = client.get(f"/api/v1/knowledge-bases/{slug}/documents")
        policies = client.get("/api/v1/policies")
        keys = client.get("/api/v1/api-keys")
        routers = client.get(f"/api/v1/routers/{slug}/versions")

    assert created.status_code == 201
    assert policy.status_code == 201
    assert any(item["slug"] == slug for item in knowledge_bases.json())
    assert knowledge_base.json()["id"] == created.json()["id"]
    assert documents.json() == []
    assert any(item["name"] == policy_name for item in policies.json())
    assert keys.json() and all(item["key"] is None for item in keys.json())
    assert routers.json() == []


def test_health_and_knowledge_base_contract() -> None:
    slug = f"test-{uuid4().hex[:10]}"
    with TestClient(app) as client:
        health = client.get("/health")
        created = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "Test", "slug": slug, "description": "API contract"},
        )
        models = client.get("/v1/models")

    assert health.json()["status"] == "ok"
    assert created.status_code == 201
    assert created.json()["slug"] == slug
    assert any(model["id"] == f"kb:{slug}:auto" for model in models.json()["data"])


def test_job_not_found_uses_typed_error_payload() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/jobs/missing")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_openai_response_includes_contextgate_metadata(monkeypatch) -> None:
    retrieval = RetrieveResponse(
        query="Can I cancel?",
        policy="balanced",
        abstained=False,
        hits=[
            RetrievalHit(
                chunk_id="orders:0",
                document_id="orders",
                source="orders.md",
                text="Orders can be cancelled before courier handoff.",
                language="en",
                score=0.9,
                rank=1,
            )
        ],
        route=RouteDecision(
            requested_policy="balanced",
            selected_policy="balanced",
            reason="explicit_policy",
            latency_budget_ms=1000,
        ),
        timings_ms={"total": 1},
        features={},
        trace_id="trace-1",
        raw_top_score=0.9,
        abstention_threshold=0.2,
    )

    class FakeAnswerUseCase:
        def execute(self, request, *, request_id=None):
            return AnswerResponse(
                answer="Orders can be cancelled before courier handoff. [1]",
                citations=[Citation(index=1, chunk_id="orders:0", source="orders.md")],
                retrieval=retrieval,
                provider="extractive",
                grounded=True,
                run_id=request_id,
                selected_provider="extractive",
                evidence_score=0.82,
                cost={"estimated_usd": 0.0012},
            )

    class FakeKnowledgeBases:
        def list_openai_models(self):
            return [{"id": "kb:demo:auto", "object": "model", "owned_by": "contextgate"}]

    class FakeContainer:
        answer_with_evidence = FakeAnswerUseCase()
        knowledge_bases = FakeKnowledgeBases()

        def startup(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(api_module, "get_container", lambda: FakeContainer())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kb:demo:auto",
                "messages": [{"role": "user", "content": "Can I cancel?"}],
            },
        )

    assert response.status_code == 200
    metadata = response.json()["contextgate"]
    assert metadata["run_id"]
    assert metadata["trace_id"] == "trace-1"
    assert metadata["selected_retrieval_policy"] == "balanced"
    assert metadata["selected_provider"] == "extractive"
    assert metadata["evidence_score"] == 0.82
    assert metadata["abstention_reason"] is None
    assert metadata["grounded"] is True
    assert metadata["cost"]["estimated_usd"] == 0.0012


def test_verified_openai_stream_emits_incremental_chunks(monkeypatch) -> None:
    retrieval = RetrieveResponse(
        query="Can I cancel?",
        policy="balanced",
        abstained=False,
        hits=[],
        route=RouteDecision(
            requested_policy="balanced",
            selected_policy="balanced",
            reason="explicit_policy",
            latency_budget_ms=1000,
        ),
        timings_ms={"total": 1},
        features={},
        trace_id="trace-stream",
        abstention_threshold=0.2,
    )

    class FakeAnswerUseCase:
        def execute(self, request, *, request_id=None):
            return AnswerResponse(
                answer="two verified tokens",
                citations=[],
                retrieval=retrieval,
                provider="extractive",
                grounded=True,
                run_id="run-stream",
                selected_provider="extractive",
            )

    class FakeContainer:
        answer_with_evidence = FakeAnswerUseCase()

        def startup(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(api_module, "get_container", lambda: FakeContainer())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kb:demo:balanced",
                "messages": [{"role": "user", "content": "Can I cancel?"}],
                "stream": True,
            },
        )

    chunks = [line for line in response.text.splitlines() if line.startswith("data: {")]
    assert len(chunks) == 4
    assert '"content": "two "' in chunks[0]
    assert '"status": "answered"' in chunks[-1]


def test_upload_is_size_limited_and_confined_to_managed_storage(monkeypatch, tmp_path) -> None:
    class FakeKnowledgeBases:
        def get(self, identifier):
            return SimpleNamespace(id="kb-id", collection_name="contextgate-demo")

    class FakeStore:
        def validate_collection_if_exists(self, collection_name):
            assert collection_name == "contextgate-demo"

    class FakeIngestionService:
        store = FakeStore()

    class FakeIngest:
        def enqueue(self, **kwargs):
            raise AssertionError("Oversized upload must not be enqueued")

    class FakeContainer:
        settings = SimpleNamespace(upload_dir=tmp_path, max_upload_bytes=4)
        knowledge_bases = FakeKnowledgeBases()
        ingest_documents = FakeIngest()
        ingestion_service = FakeIngestionService()

        def startup(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(api_module, "get_container", lambda: FakeContainer())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/knowledge-bases/demo/documents",
            files={"file": ("../../escape.md", b"too-large", "text/markdown")},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert not list(tmp_path.rglob("escape.md"))


def test_upload_rejects_incompatible_embedding_schema_before_writing(monkeypatch, tmp_path) -> None:
    class FakeKnowledgeBases:
        def get(self, identifier):
            return SimpleNamespace(id="kb-id", collection_name="contextgate-demo")

    class FakeStore:
        def validate_collection_if_exists(self, collection_name):
            raise ValueError(
                f"Incompatible Qdrant collection {collection_name}: dense dimension is 384, "
                "expected 64"
            )

    class FakeContainer:
        settings = SimpleNamespace(upload_dir=tmp_path, max_upload_bytes=100)
        knowledge_bases = FakeKnowledgeBases()
        ingestion_service = SimpleNamespace(store=FakeStore())

        def startup(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(api_module, "get_container", lambda: FakeContainer())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/knowledge-bases/demo/documents",
            files={"file": ("policy.md", b"Policy text", "text/markdown")},
        )

    assert response.status_code == 422
    assert "embedding schema" in response.json()["message"]
    assert "create a new knowledge base" in response.json()["details"]["remedy"]
    assert not list(tmp_path.rglob("policy.md"))


def test_upload_rejects_oversized_idempotency_key(monkeypatch, tmp_path) -> None:
    class FakeKnowledgeBases:
        def get(self, identifier):
            return SimpleNamespace(id="kb-id", collection_name="contextgate-demo")

    class FakeStore:
        def validate_collection_if_exists(self, collection_name):
            return None

    class FakeIngest:
        def enqueue(self, **kwargs):
            raise AssertionError("Invalid idempotency key must not be persisted")

    class FakeContainer:
        settings = SimpleNamespace(upload_dir=tmp_path, max_upload_bytes=100)
        knowledge_bases = FakeKnowledgeBases()
        ingestion_service = SimpleNamespace(store=FakeStore())
        ingest_documents = FakeIngest()

        def startup(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(api_module, "get_container", lambda: FakeContainer())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/knowledge-bases/demo/documents",
            headers={"Idempotency-Key": "x" * 129},
            files={"file": ("policy.md", b"Policy text", "text/markdown")},
        )

    assert response.status_code == 422
    assert response.json()["details"]["max_length"] == 128
    assert not list(tmp_path.rglob("policy.md"))
