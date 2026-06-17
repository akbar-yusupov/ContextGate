from __future__ import annotations

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
        "OpenAI Compatibility",
    ]
    assert schema["paths"]["/api/v1/runs/answer"]["post"]["tags"] == ["Answer Runtime"]
    assert schema["paths"]["/v1/chat/completions"]["post"]["tags"] == ["OpenAI Compatibility"]


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
