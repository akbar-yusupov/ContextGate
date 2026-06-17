from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from contextgate.adapters.qdrant.vector_index import StoreHit
from contextgate.application.dto import RetrieveCommand
from contextgate.application.retrieval import RetrievalService
from contextgate.config import PoliciesConfig, PolicyConfig, Settings
from contextgate.domain.retrieval import RouteDecision


class FixedPolicyVectorIndex:
    def __init__(self) -> None:
        self.settings = Settings(
            environment="test",
            embedding_backend="deterministic",
            dense_dimension=64,
            late_dimension=32,
        )
        self.policy_search_calls = 0

    def policy_search(self, *args: Any, **kwargs: Any) -> list[StoreHit]:
        self.policy_search_calls += 1
        return [
            StoreHit(
                chunk_id="orders:0",
                document_id="orders",
                source="orders.md",
                text="Cancel before courier handoff.",
                language="en",
                score=0.9,
                metadata={},
            )
        ]

    def embed_query(self, *args: Any, **kwargs: Any) -> object:
        raise AssertionError("Explicit policies must not execute the router probe")

    def probe_search(self, *args: Any, **kwargs: Any) -> tuple[list[StoreHit], list[StoreHit]]:
        raise AssertionError("Explicit policies must not execute the router probe")


class FakeRouter:
    def decide(
        self,
        knowledge_base: str,
        features: dict[str, float | int | str],
        latency_budget_ms: float,
    ) -> RouteDecision:
        raise AssertionError("Explicit policies must not call the router")

    def abstention_threshold(
        self,
        knowledge_base: str,
        policy: str,
        *,
        fallback: float,
    ) -> float:
        return fallback


class FakeKnowledgeBases:
    def get(self, identifier: str) -> SimpleNamespace:
        return SimpleNamespace(slug=identifier, collection_name=identifier)


def test_explicit_policy_skips_router_probe() -> None:
    vector_index = FixedPolicyVectorIndex()
    policy = PolicyConfig(
        dense_limit=20,
        sparse_limit=0,
        prefetch_limit=20,
        output_limit=10,
        use_late_interaction=False,
        use_cross_encoder=False,
        abstention_threshold=0.2,
    )
    service = RetrievalService(
        vector_index=vector_index,
        policies=PoliciesConfig(policies={"fast": policy, "balanced": policy, "accurate": policy}),
        router=FakeRouter(),
        knowledge_bases=FakeKnowledgeBases(),
    )

    response = service.retrieve(
        RetrieveCommand(
            knowledge_base="demo",
            query="How can I cancel an order?",
            policy="fast",
        )
    )

    assert vector_index.policy_search_calls == 1
    assert response.features["first_stage_latency_ms"] == 0
    assert response.raw_top_score == 0.9
