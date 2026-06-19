import sys
from types import SimpleNamespace

import pytest

from contextgate.adapters.litellm.generation import AnswerGenerator, _completion
from contextgate.config import Settings
from contextgate.domain.errors import ContextGateError
from contextgate.domain.retrieval import RetrievalHit, RetrievalResult, RouteDecision


def test_missing_litellm_extra_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(ContextGateError, match="Install ContextGate with the 'llm' extra"):
        _completion(model="openai/test")


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


def test_context_is_packed_to_the_declared_token_budget() -> None:
    retrieval = RetrievalResult(
        query="What is the policy?",
        policy="fast",
        abstained=False,
        hits=[
            RetrievalHit(
                chunk_id="long:0",
                document_id="long",
                source="long.md",
                text="x" * 200,
                language="en",
                score=0.9,
                rank=1,
            )
        ],
        route=RouteDecision(
            requested_policy="fast",
            selected_policy="fast",
            reason="explicit_policy",
        ),
        timings_ms={"total": 1},
        features={},
        trace_id="trace",
        raw_top_score=0.9,
        abstention_threshold=0.2,
    )

    response = AnswerGenerator(Settings(llm_model=None)).generate(
        retrieval,
        max_context_tokens=5,
    )

    assert len(response.retrieval.hits[0].text) == 20


def test_provider_usage_and_configured_pricing_are_recorded(monkeypatch) -> None:
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
            )
        ],
        route=RouteDecision(
            requested_policy="fast",
            selected_policy="fast",
            reason="explicit_policy",
        ),
        timings_ms={"total": 1},
        features={},
        trace_id="trace",
        raw_top_score=0.9,
        abstention_threshold=0.2,
    )

    def completion(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Orders can be cancelled before courier handoff. [1]"
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    generator = AnswerGenerator(
        Settings(
            llm_model="openai/test",
            llm_input_cost_per_1m_tokens=1.0,
            llm_output_cost_per_1m_tokens=2.0,
        )
    )

    response = generator.generate(retrieval, provider="openai/test")

    assert response.cost["input_tokens"] == 100
    assert response.cost["output_tokens"] == 20
    assert response.cost["actual_usd"] == 0.00014


def test_provisional_generation_forwards_real_provider_deltas(monkeypatch) -> None:
    retrieval = RetrievalResult(
        query="When can I cancel?",
        policy="fast",
        abstained=False,
        hits=[
            RetrievalHit(
                chunk_id="orders:0",
                document_id="orders",
                source="orders.md",
                text="Orders can be cancelled.",
                language="en",
                score=0.9,
                rank=1,
            )
        ],
        route=RouteDecision(
            requested_policy="fast",
            selected_policy="fast",
            reason="explicit_policy",
        ),
        timings_ms={"total": 1},
        features={},
        trace_id="trace",
        raw_top_score=0.9,
        abstention_threshold=0.2,
    )

    def completion(**kwargs):
        assert kwargs["stream"] is True
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Orders "))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="can be cancelled. [1]"))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            ),
        ]

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    deltas: list[str] = []
    response = AnswerGenerator(Settings(llm_model="openai/test")).generate(
        retrieval,
        provider="openai/test",
        on_token=deltas.append,
    )

    assert deltas == ["Orders ", "can be cancelled. [1]"]
    assert response.answer == "Orders can be cancelled. [1]"


def test_repeated_provider_failure_opens_circuit(monkeypatch) -> None:
    retrieval = RetrievalResult(
        query="When can I cancel?",
        policy="fast",
        abstained=False,
        hits=[
            RetrievalHit(
                chunk_id="orders:0",
                document_id="orders",
                source="orders.md",
                text="Orders can be cancelled.",
                language="en",
                score=0.9,
                rank=1,
            )
        ],
        route=RouteDecision(
            requested_policy="fast",
            selected_policy="fast",
            reason="explicit_policy",
        ),
        timings_ms={"total": 1},
        features={},
        trace_id="trace",
        raw_top_score=0.9,
        abstention_threshold=0.2,
    )
    calls = 0

    def completion(**kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("provider timeout")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    generator = AnswerGenerator(
        Settings(
            llm_model="openai/test",
            llm_circuit_failure_threshold=1,
            llm_circuit_cooldown_seconds=60,
        )
    )

    with pytest.raises(ContextGateError, match="call failed"):
        generator.generate(retrieval, provider="openai/test")
    with pytest.raises(ContextGateError, match="circuit is open"):
        generator.generate(retrieval, provider="openai/test")

    assert calls == 1
