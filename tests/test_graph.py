from __future__ import annotations

from typing import Any

from contextgate.adapters.langgraph.runtime import GatewayGraph
from contextgate.application.dto import AnswerCommand
from contextgate.domain.gateway import AnswerResult, Citation
from contextgate.domain.retrieval import RetrievalHit, RetrievalResult, RouteDecision


def _route() -> RouteDecision:
    return RouteDecision(
        requested_policy="auto",
        selected_policy="balanced",
        reason="test",
        latency_budget_ms=1000,
    )


def _retrieval_result(
    *, abstained: bool, hits: list[RetrievalHit] | None = None
) -> RetrievalResult:
    return RetrievalResult(
        query="Can I cancel the order?",
        policy="balanced",
        abstained=abstained,
        hits=hits or [],
        route=_route(),
        timings_ms={"total": 1},
        features={},
        trace_id="trace",
        raw_top_score=0.9 if hits else None,
        abstention_threshold=0.2,
    )


class AbstainingRetrieval:
    def retrieve(self, request: AnswerCommand) -> RetrievalResult:
        return _retrieval_result(abstained=True)


class CountingGenerator:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.abstain_calls = 0

    def generate(
        self, retrieval: RetrievalResult, *, system_prompt: str | None = None
    ) -> AnswerResult:
        self.generate_calls += 1
        raise AssertionError("Generation must be skipped when evidence is insufficient")

    def abstain(self, retrieval: RetrievalResult) -> AnswerResult:
        self.abstain_calls += 1
        return AnswerResult(
            answer="Insufficient evidence.",
            citations=[],
            retrieval=retrieval,
            provider="abstention",
            grounded=False,
        )


def test_graph_routes_abstention_without_generation() -> None:
    generator = CountingGenerator()
    graph = GatewayGraph(
        retrieval=AbstainingRetrieval(),  # type: ignore[arg-type]
        generator=generator,  # type: ignore[arg-type]
    )
    compiled = graph.graph

    response = graph.answer(AnswerCommand(knowledge_base="demo", query="unknown"))

    assert graph.graph is compiled
    assert response.provider == "abstention"
    assert response.grounded is False
    assert response.abstention_reason == "retrieval_empty"
    assert generator.abstain_calls == 1
    assert generator.generate_calls == 0


class LowCoverageRetrieval:
    def retrieve(self, request: AnswerCommand) -> RetrievalResult:
        return _retrieval_result(
            abstained=False,
            hits=[
                RetrievalHit(
                    chunk_id="billing:0",
                    document_id="billing",
                    source="billing.md",
                    text="Invoices are generated after payment and appear in Billing.",
                    language="en",
                    score=0.9,
                    rank=1,
                    metadata={},
                )
            ],
        )


def test_graph_abstains_on_low_coverage_without_generation() -> None:
    generator = CountingGenerator()
    graph = GatewayGraph(
        retrieval=LowCoverageRetrieval(),  # type: ignore[arg-type]
        generator=generator,  # type: ignore[arg-type]
    )

    response = graph.answer(
        AnswerCommand(
            knowledge_base="demo",
            query="Can I rent a spaceship with cryptocurrency?",
        )
    )

    assert response.provider == "abstention"
    assert response.grounded is False
    assert response.abstention_reason == "low_coverage"
    assert generator.abstain_calls == 1
    assert generator.generate_calls == 0


class SupportedRetrieval:
    def retrieve(self, request: AnswerCommand) -> RetrievalResult:
        return _retrieval_result(
            abstained=False,
            hits=[
                RetrievalHit(
                    chunk_id="orders:0",
                    document_id="orders",
                    source="orders.md",
                    text="You can cancel the order before courier handoff.",
                    language="en",
                    score=0.9,
                    rank=1,
                    metadata={},
                )
            ],
        )


class InvalidCitationGenerator:
    def generate(
        self,
        retrieval: RetrievalResult,
        *,
        system_prompt: str | None = None,
    ) -> AnswerResult:
        return AnswerResult(
            answer="You can cancel the order before courier handoff. [1]",
            citations=[Citation(index=1, chunk_id="wrong-chunk", source="orders.md")],
            retrieval=retrieval,
            provider="test-llm",
            grounded=True,
        )

    def abstain(self, retrieval: RetrievalResult) -> AnswerResult:
        raise AssertionError("Evidence should be sufficient for generation")


def test_graph_marks_invalid_citation_as_ungrounded() -> None:
    graph = GatewayGraph(
        retrieval=SupportedRetrieval(),  # type: ignore[arg-type]
        generator=InvalidCitationGenerator(),  # type: ignore[arg-type]
    )

    response = graph.answer(
        AnswerCommand(knowledge_base="demo", query="Can I cancel the order?", policy="balanced")
    )

    assert response.grounded is False
    assert response.abstention_reason == "invalid_citations"
    assert response.evidence_score > 0


def test_streaming_events_are_in_deterministic_gateway_order() -> None:
    graph = GatewayGraph(
        retrieval=SupportedRetrieval(),  # type: ignore[arg-type]
        generator=InvalidCitationGenerator(),  # type: ignore[arg-type]
    )

    events: list[dict[str, Any]] = graph.stream_events(
        AnswerCommand(knowledge_base="demo", query="Can I cancel the order?", policy="balanced")
    )

    assert [event["event"] for event in events] == [
        "query_analyzed",
        "retrieval_started",
        "evidence_scored",
        "provider_selected",
        "citation_verified",
        "final",
    ]
