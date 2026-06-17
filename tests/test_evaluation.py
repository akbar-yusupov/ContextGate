from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from contextgate.adapters.mlflow.evaluation_store import (
    BenchmarkService,
    summarize_gateway_cases,
)
from contextgate.config import Settings
from contextgate.domain.evaluation import ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from contextgate.domain.gateway import AbstentionReason, AnswerResult, Citation
from contextgate.domain.retrieval import RetrievalHit, RetrievalResult, RouteDecision


def test_retrieval_metrics() -> None:
    retrieved = ["x", "b", "a"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(0.6934, abs=0.001)


def test_unanswerable_query_rewards_abstention() -> None:
    assert recall_at_k([], set(), 10) == 1
    assert ndcg_at_k([], set(), 10) == 1
    assert recall_at_k(["hallucination"], set(), 10) == 0


def test_percentile_interpolates() -> None:
    assert percentile([10, 20, 30, 40], 0.5) == 25


def _route() -> RouteDecision:
    return RouteDecision(
        requested_policy="balanced",
        selected_policy="balanced",
        reason="test",
        latency_budget_ms=1000,
    )


def _retrieval(
    query: str,
    *,
    hits: list[RetrievalHit] | None = None,
    abstained: bool = False,
) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        policy="balanced",
        abstained=abstained,
        hits=hits or [],
        route=_route(),
        timings_ms={"total": 1.0},
        features={"language": "en", "query_token_count": len(query.split())},
        trace_id=f"trace-{query}",
        raw_top_score=0.9 if hits else None,
        abstention_threshold=0.2,
    )


def _hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="refunds:0",
        document_id="refunds",
        source="refunds.md",
        text="Refunds are available within 14 days.",
        language="en",
        score=0.9,
        rank=1,
        metadata={},
    )


class FakeKnowledgeBases:
    def get(self, knowledge_base: str) -> SimpleNamespace:
        return SimpleNamespace(collection_name=f"contextgate-{knowledge_base}")


class FakeRetrieval:
    def __init__(self) -> None:
        self.knowledge_bases = FakeKnowledgeBases()
        self.direct_retrieve_calls = 0

    def probe(self, collection_name: str, query: str, limit: int = 20) -> SimpleNamespace:
        return SimpleNamespace(features={"first_stage_latency_ms": 1.0})

    def retrieve(self, request) -> RetrievalResult:
        self.direct_retrieve_calls += 1
        raise AssertionError("Gateway evaluation must use AnswerWithEvidence")


class FakeAnswerGateway:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, request, *, request_id=None) -> AnswerResult:
        self.calls.append((request, request_id))
        if "refund" in request.query.lower():
            hit = _hit()
            return AnswerResult(
                answer="Refunds are available within 14 days. [1]",
                citations=[Citation(index=1, chunk_id=hit.chunk_id, source=hit.source)],
                retrieval=_retrieval(request.query, hits=[hit]),
                provider="extractive",
                selected_provider="extractive",
                grounded=True,
                run_id=request_id,
                evidence_score=0.9,
                answerability_score=0.9,
                coverage_score=0.9,
                support_score=0.9,
                cost={"estimated_usd": 0.0},
            )
        return AnswerResult(
            answer="I could not answer from grounded evidence in the knowledge base.",
            citations=[],
            retrieval=_retrieval(request.query, abstained=True),
            provider="abstention",
            selected_provider="abstention",
            grounded=False,
            run_id=request_id,
            evidence_score=0.0,
            abstention_reason=AbstentionReason.RETRIEVAL_EMPTY,
            cost={"estimated_usd": 0.0},
        )


def test_gateway_evaluation_uses_answer_with_evidence(tmp_path) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "query": "How do refunds work?",
                        "language": "en",
                        "relevant_chunk_ids": ["refunds:0"],
                        "expected_facts": ["Refunds are available within 14 days."],
                        "answerable": True,
                        "tags": ["refunds"],
                    }
                ),
                json.dumps(
                    {
                        "id": "q2",
                        "query": "Do you accept cryptocurrency?",
                        "language": "en",
                        "relevant_chunk_ids": [],
                        "expected_facts": [],
                        "answerable": False,
                        "tags": ["unanswerable"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    retrieval = FakeRetrieval()
    gateway = FakeAnswerGateway()
    service = BenchmarkService(
        retrieval=retrieval,  # type: ignore[arg-type]
        settings=Settings(
            report_dir=tmp_path / "reports",
            mlflow_tracking_uri=str(tmp_path / "mlruns"),
            embedding_backend="deterministic",
        ),
        answer_gateway=gateway,
    )

    result = service.run(
        session=object(),  # type: ignore[arg-type]
        knowledge_base="demo",
        dataset_path=dataset,
        policies=["balanced"],
        evaluate_answers=True,
    )

    assert retrieval.direct_retrieve_calls == 0
    assert len(gateway.calls) == 2
    assert all(call[0].debug for call in gateway.calls)
    assert result["gateway_summary"]["overall"]["answer_rate"] == 0.5
    assert result["gateway_summary"]["overall"]["correct_abstention_rate"] == 1.0
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "QA Gate Summary" in report
    payload = json.loads(Path(result["results_path"]).read_text(encoding="utf-8"))
    assert payload["gateway_evaluation"]["cases"][0]["run_id"].startswith("eval-")
    assert payload["gateway_evaluation"]["cases"][0]["trace_id"]


def test_gateway_summary_counts_false_answers_and_false_abstentions() -> None:
    cases = [
        {
            "policy": "balanced",
            "answerable": False,
            "answered": True,
            "abstained": False,
            "grounded": True,
            "failure_type": "false_answer",
            "citation_validity": 1.0,
            "unsupported_claim_count": 0,
            "latency_ms": 10.0,
            "cost_estimated_usd": 0.002,
        },
        {
            "policy": "balanced",
            "answerable": True,
            "answered": False,
            "abstained": True,
            "grounded": False,
            "failure_type": "false_abstention",
            "citation_validity": 1.0,
            "unsupported_claim_count": 1,
            "latency_ms": 20.0,
            "cost_estimated_usd": 0.0,
        },
    ]

    summary = summarize_gateway_cases(cases)["overall"]

    assert summary["false_answer_rate"] == 1.0
    assert summary["false_abstention_rate"] == 1.0
    assert summary["unsupported_claim_case_count"] == 1.0
    assert summary["estimated_cost_per_answer"] == 0.002
