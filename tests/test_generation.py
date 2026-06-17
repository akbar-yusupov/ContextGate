from contextgate.adapters.litellm.generation import AnswerGenerator
from contextgate.config import Settings
from contextgate.domain.retrieval import RetrievalHit, RetrievalResult, RouteDecision


def test_extractive_generation_keeps_citations() -> None:
    retrieval = RetrievalResult(
        query="When can I cancel?",
        policy="fast",
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
                metadata={},
            )
        ],
        route=RouteDecision(
            requested_policy="fast",
            selected_policy="fast",
            reason="explicit_policy",
            latency_budget_ms=1000,
        ),
        timings_ms={"total": 1},
        features={},
        trace_id="trace",
        raw_top_score=0.9,
        abstention_threshold=0.2,
    )
    generator = AnswerGenerator(Settings(llm_model=None))

    response = generator.generate(retrieval)

    assert response.provider == "extractive"
    assert response.grounded is True
    assert response.citations[0].chunk_id == "orders:0"
    assert "[1]" in response.answer


def test_abstention_does_not_leak_retrieved_context() -> None:
    retrieval = RetrievalResult(
        query="Can I rent a spaceship?",
        policy="fast",
        abstained=False,
        hits=[
            RetrievalHit(
                chunk_id="billing:0",
                document_id="billing",
                source="billing.md",
                text="Invoices are generated after payment.",
                language="en",
                score=0.9,
                rank=1,
                metadata={},
            )
        ],
        route=RouteDecision(
            requested_policy="fast",
            selected_policy="fast",
            reason="explicit_policy",
            latency_budget_ms=1000,
        ),
        timings_ms={"total": 1},
        features={},
        trace_id="trace",
        raw_top_score=0.9,
        abstention_threshold=0.2,
    )
    generator = AnswerGenerator(Settings(llm_model=None))

    response = generator.abstain(retrieval)

    assert response.provider == "abstention"
    assert response.grounded is False
    assert response.citations == []
    assert "Invoices" not in response.answer
