from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from contextgate.application.dto import AnswerCommand
from contextgate.application.use_cases import AnswerWithEvidence
from contextgate.domain.gateway import AbstentionReason, AnswerResult
from contextgate.domain.models import CostRecord
from contextgate.domain.retrieval import RetrievalHit, RetrievalResult, RouteDecision


def _route() -> RouteDecision:
    return RouteDecision(
        requested_policy="balanced",
        selected_policy="balanced",
        reason="test",
        latency_budget_ms=1000,
    )


def _retrieval(*, abstained: bool, contexts: list[str] | None = None) -> RetrievalResult:
    hits = [
        RetrievalHit(
            chunk_id=f"chunk-{index}",
            document_id="doc",
            source="doc.md",
            text=text,
            language="en",
            score=0.9,
            rank=index + 1,
            metadata={},
        )
        for index, text in enumerate(contexts or [])
    ]
    return RetrievalResult(
        query="Can I cancel my order?",
        policy="balanced",
        abstained=abstained,
        hits=hits,
        route=_route(),
        timings_ms={"total": 1.0},
        features={"language": "en", "query_token_count": 5},
        trace_id="trace-1",
        raw_top_score=0.9 if hits else None,
        abstention_threshold=0.2,
    )


@dataclass
class FakeGraph:
    result: AnswerResult

    def answer(self, request: AnswerCommand) -> AnswerResult:
        return self.result


class RecordingProviderRegistry:
    def __init__(self) -> None:
        self.calls = 0

    def choose(self, **_: Any) -> str:
        self.calls += 1
        return "openai/gpt-4o-mini"

    def list(self) -> list[dict[str, Any]]:
        return []

    def test(self, provider: str | None = None) -> dict[str, Any]:
        return {"provider": provider, "available": True}


class RecordingCostLedger:
    def __init__(self) -> None:
        self.records: list[CostRecord] = []

    def record(self, record: CostRecord) -> None:
        self.records.append(record)

    def list_for_run(self, run_id: str) -> list[CostRecord]:
        return [record for record in self.records if record.run_id == run_id]


class RecordingTraceStore:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []
        self.started: list[tuple[str, str]] = []

    def start_run(
        self,
        run_id: str,
        *,
        correlation_id: str,
        knowledge_base: str,
        query: str,
    ) -> None:
        self.started.append((run_id, correlation_id))

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((run_id, event_type, payload))

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        return []

    def get_trace(self, run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "events": []}

    def purge_older_than(self, days: int) -> int:
        return 0


class FakeUnitOfWork:
    raw_session: Any = None

    def __enter__(self) -> FakeUnitOfWork:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def create_job(self, **_: Any) -> tuple[Any, bool]:
        raise NotImplementedError

    def promote_router_version(self, run_id: str, knowledge_base: str) -> None:
        raise NotImplementedError

    def mark_job_enqueued(self, job_id: str) -> None:
        raise NotImplementedError

    def pending_job_dispatches(self) -> list[tuple[str, str]]:
        return []


class FakeUnitOfWorkFactory:
    def __call__(self) -> FakeUnitOfWork:
        return FakeUnitOfWork()


class DictCache:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


def _use_case(
    result: AnswerResult,
) -> tuple[AnswerWithEvidence, RecordingProviderRegistry, RecordingCostLedger]:
    provider_registry = RecordingProviderRegistry()
    cost_ledger = RecordingCostLedger()
    use_case = AnswerWithEvidence(
        FakeGraph(result),
        FakeUnitOfWorkFactory(),
        provider_registry,
        cost_ledger,
        RecordingTraceStore(),
        DictCache(),
    )
    return use_case, provider_registry, cost_ledger


def test_answer_use_case_records_abstention_without_provider_routing() -> None:
    result = AnswerResult(
        answer="I could not find enough grounded evidence in the knowledge base.",
        citations=[],
        retrieval=_retrieval(abstained=True),
        provider="abstention",
        grounded=False,
        abstention_reason=AbstentionReason.RETRIEVAL_EMPTY,
    )
    use_case, provider_registry, cost_ledger = _use_case(result)

    response = use_case.execute(
        AnswerCommand(knowledge_base="demo", query="Can I cancel my order?"),
        request_id="run-1",
    )

    assert response.selected_provider == "abstention"
    assert response.abstention_reason == AbstentionReason.RETRIEVAL_EMPTY
    assert response.grounded is False
    assert provider_registry.calls == 0
    assert cost_ledger.records[0].provider == "abstention"


def test_answer_use_case_surfaces_unsupported_claims() -> None:
    result = AnswerResult(
        answer="You can cancel the order and receive an instant teleport refund.",
        citations=[],
        retrieval=_retrieval(
            abstained=False,
            contexts=["Orders can be cancelled before courier handoff."],
        ),
        provider="extractive",
        grounded=True,
    )
    use_case, provider_registry, cost_ledger = _use_case(result)

    response = use_case.execute(
        AnswerCommand(knowledge_base="demo", query="Can I cancel my order?"),
        request_id="run-2",
    )

    assert "teleport" in response.unsupported_claims
    assert "teleport" not in response.answer
    assert response.provider == "abstention"
    assert response.selected_provider == "abstention"
    assert response.abstention_reason == AbstentionReason.INVALID_CITATIONS
    assert response.grounded is False
    assert provider_registry.calls == 0
    assert cost_ledger.records[0].provider == "abstention"


def test_client_correlation_id_never_becomes_the_run_primary_key() -> None:
    result = AnswerResult(
        answer="",
        citations=[],
        retrieval=_retrieval(abstained=True),
        provider="abstention",
        grounded=False,
        abstention_reason=AbstentionReason.RETRIEVAL_EMPTY,
    )
    use_case, _, _ = _use_case(result)

    first = use_case.execute(
        AnswerCommand(knowledge_base="demo", query="unknown"), request_id="client-id"
    )
    second = use_case.execute(
        AnswerCommand(knowledge_base="demo", query="unknown"), request_id="client-id"
    )

    assert first.run_id != "client-id"
    assert second.run_id != "client-id"
    assert first.run_id != second.run_id


def test_promoted_policy_is_applied_as_an_immutable_run_snapshot() -> None:
    result = AnswerResult(
        answer="",
        citations=[],
        retrieval=_retrieval(abstained=True),
        provider="abstention",
        grounded=False,
        abstention_reason=AbstentionReason.RETRIEVAL_EMPTY,
    )

    class CapturingGraph(FakeGraph):
        request: AnswerCommand | None = None

        def answer(self, request: AnswerCommand) -> AnswerResult:
            self.request = request
            return self.result

    class PolicyRepository:
        def resolve_active(self, policy_id: str):
            assert policy_id == "policy-1"
            return SimpleNamespace(
                id="policy-1",
                name="strict",
                status="active",
                retrieval_policy="accurate",
                provider_policy="extractive",
                latency_budget_ms=750,
                cost_budget_usd=0.0,
                promoted_at=datetime.now(UTC),
            )

    graph = CapturingGraph(result)
    use_case = AnswerWithEvidence(
        graph,
        FakeUnitOfWorkFactory(),
        RecordingProviderRegistry(),
        RecordingCostLedger(),
        RecordingTraceStore(),
        DictCache(),
        PolicyRepository(),
    )

    response = use_case.execute(
        AnswerCommand(
            knowledge_base="demo",
            query="unknown",
            gateway_policy_id="policy-1",
        )
    )

    assert graph.request is not None
    assert graph.request.policy == "accurate"
    assert graph.request.llm_provider == "extractive"
    assert graph.request.latency_budget_ms == 750
    assert response.policy_snapshot["id"] == "policy-1"
