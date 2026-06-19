from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import time
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

import mlflow
import pandas as pd
from mlflow.data import from_pandas
from sqlalchemy.orm import Session

from contextgate.adapters.mlflow.reporting import write_html_report
from contextgate.application.dto import AnswerCommand, RetrieveCommand
from contextgate.application.retrieval import RetrievalService
from contextgate.config import Settings, get_settings
from contextgate.domain.evaluation import (
    BenchmarkQuery,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from contextgate.domain.gateway import AnswerResult
from contextgate.domain.retrieval import PolicyName


class AnswerGateway(Protocol):
    def execute(
        self,
        request: AnswerCommand,
        *,
        request_id: str | None = None,
    ) -> AnswerResult: ...


def load_benchmark(path: Path) -> list[BenchmarkQuery]:
    queries: list[BenchmarkQuery] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                queries.append(BenchmarkQuery.from_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid benchmark line {line_number}: {exc}") from exc
    return queries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def _token_recall(expected: str, actual: str) -> float:
    expected_tokens = {token for token in re.findall(r"\w+", expected.lower()) if len(token) > 2}
    actual_tokens = set(re.findall(r"\w+", actual.lower()))
    if not expected_tokens:
        return 1.0
    return len(expected_tokens & actual_tokens) / len(expected_tokens)


def _lexical_faithfulness(answer: str, contexts: list[str]) -> float:
    answer_tokens = {token for token in re.findall(r"\w+", answer.lower()) if len(token) > 2}
    context_tokens = set(re.findall(r"\w+", " ".join(contexts).lower()))
    if not answer_tokens:
        return 1.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def _is_abstained(answer: AnswerResult) -> bool:
    return (
        answer.abstention_reason is not None
        or answer.provider == "abstention"
        or answer.selected_provider == "abstention"
        or answer.retrieval.abstained
    )


def _citation_validity(answer: AnswerResult) -> float:
    if _is_abstained(answer):
        return 1.0
    return float(answer.grounded and bool(answer.citations))


def _gateway_failure_type(
    *,
    query: BenchmarkQuery,
    answer: AnswerResult,
    fact_coverage: float,
    citation_validity: float,
) -> str:
    abstained = _is_abstained(answer)
    if not query.answerable:
        if not abstained and answer.grounded:
            return "false_answer"
        if not abstained and citation_validity < 1:
            return "invalid_citation"
        return "ok"
    if abstained:
        return "false_abstention"
    if citation_validity < 1:
        return "invalid_citation"
    if query.expected_facts and fact_coverage < 0.5:
        return "low_fact_coverage"
    return "ok"


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _wilson_bound(successes: int, total: int, *, upper: bool) -> float:
    if total == 0:
        return 1.0 if upper else 0.0
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total) / denominator
    return min(1.0, center + margin) if upper else max(0.0, center - margin)


def summarize_gateway_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, float]:
        total = len(items)
        answerable = [item for item in items if item["answerable"]]
        unanswerable = [item for item in items if not item["answerable"]]
        answered = [item for item in items if item["answered"]]
        abstained = [item for item in items if item["abstained"]]
        grounded_answers = [item for item in answered if item["grounded"]]
        unsupported_claim_cases = [item for item in items if item["unsupported_claim_count"] > 0]
        contradiction_cases = [
            item for item in items if item.get("contradiction_claim_count", 0) > 0
        ]
        supported_answers = [
            item for item in answered if item["unsupported_claim_count"] == 0 and item["grounded"]
        ]
        false_answers = sum(1 for item in unanswerable if item["failure_type"] == "false_answer")
        valid_citations = sum(float(item["citation_validity"]) >= 1 for item in answered)
        return {
            "case_count": float(total),
            "answer_rate": _rate(len(answered), total),
            "abstention_rate": _rate(len(abstained), total),
            "correct_abstention_rate": _rate(
                sum(1 for item in unanswerable if item["abstained"]),
                len(unanswerable),
            ),
            "false_answer_rate": _rate(
                sum(1 for item in unanswerable if item["failure_type"] == "false_answer"),
                len(unanswerable),
            ),
            "false_abstention_rate": _rate(
                sum(1 for item in answerable if item["failure_type"] == "false_abstention"),
                len(answerable),
            ),
            "grounded_answer_rate": _rate(len(grounded_answers), len(answered)),
            "citation_validity_rate": mean([float(item["citation_validity"]) for item in items])
            if items
            else 0.0,
            "unsupported_claim_case_count": float(len(unsupported_claim_cases)),
            "contradiction_case_count": float(len(contradiction_cases)),
            "claim_support_rate": _rate(len(supported_answers), len(answered)),
            "false_answer_upper_95": _wilson_bound(false_answers, len(unanswerable), upper=True),
            "citation_validity_lower_95": _wilson_bound(
                valid_citations, len(answered), upper=False
            ),
            "claim_support_lower_95": _wilson_bound(
                len(supported_answers), len(answered), upper=False
            ),
            "latency_p50_ms": percentile([float(item["latency_ms"]) for item in items], 0.50),
            "latency_p95_ms": percentile([float(item["latency_ms"]) for item in items], 0.95),
            "estimated_cost_per_answer": _rate(
                sum(float(item["cost_estimated_usd"]) for item in items),
                len(answered),
            ),
        }

    by_policy: dict[str, dict[str, float]] = {}
    for policy in sorted({str(item["policy"]) for item in cases}):
        by_policy[policy] = summarize([item for item in cases if item["policy"] == policy])
    return {"overall": summarize(cases), "by_policy": by_policy}


class BenchmarkService:
    def __init__(
        self,
        retrieval: RetrievalService,
        settings: Settings | None = None,
        answer_gateway: AnswerGateway | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.settings = settings or get_settings()
        self.answer_gateway = answer_gateway

    def run(
        self,
        session: Session,
        knowledge_base: str,
        dataset_path: Path,
        policies: list[PolicyName] | None = None,
        evaluate_answers: bool = False,
    ) -> dict[str, Any]:
        policies = policies or ["fast", "balanced", "accurate"]
        if evaluate_answers and self.answer_gateway is None:
            raise ValueError("Gateway answer evaluation requires an AnswerWithEvidence use case")
        queries = load_benchmark(dataset_path)
        if not queries:
            raise ValueError("Benchmark dataset is empty")
        mlflow.set_tracking_uri(self.settings.resolved_mlflow_tracking_uri)
        mlflow.set_experiment("contextgate-benchmarks")
        dataset_digest = _sha256(dataset_path)
        with mlflow.start_run(run_name=f"{knowledge_base}-{dataset_path.stem}") as run:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The specified dataset source can be interpreted in multiple ways.*",
                    category=UserWarning,
                    module="mlflow.data.dataset_source_registry",
                )
                dataset = from_pandas(
                    pd.DataFrame([query.to_dict() for query in queries]),
                    source=str(dataset_path.resolve()),
                    name=dataset_path.stem,
                    digest=dataset_digest[:36],
                )
            mlflow.log_input(dataset, context="retrieval-evaluation")
            run_id = run.info.run_id
            output_dir = self.settings.report_dir / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, Any]] = []
            aggregates: dict[str, dict[str, list[float]]] = {
                policy: defaultdict(list) for policy in policies
            }
            language_scores: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            tag_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            gateway_cases: list[dict[str, Any]] = []
            for query in queries:
                kb = self.retrieval.knowledge_bases.get(knowledge_base)
                probe = self.retrieval.probe(kb.collection_name, query.query, limit=20)
                row: dict[str, Any] = {
                    "id": query.id,
                    "group_id": query.group_id or query.id,
                    "query": query.query,
                    "language": query.language,
                    "tags": query.tags,
                    "answerable": query.answerable,
                    "features": probe.features,
                    "probe_latency_ms": float(probe.features["first_stage_latency_ms"]),
                    "policies": {},
                }
                relevant = set(query.relevant_chunk_ids)
                for policy in policies:
                    gateway_answer: AnswerResult | None = None
                    gateway_latency_ms: float | None = None
                    command = RetrieveCommand(
                        knowledge_base=knowledge_base,
                        query=query.query,
                        policy=policy,
                        latency_budget_ms=10_000,
                        limit=10,
                    )
                    if evaluate_answers:
                        assert self.answer_gateway is not None
                        answer_command = AnswerCommand(
                            knowledge_base=knowledge_base,
                            query=query.query,
                            policy=policy,
                            latency_budget_ms=10_000,
                            limit=10,
                            debug=True,
                        )
                        started_at = time.perf_counter()
                        gateway_answer = self.answer_gateway.execute(
                            answer_command,
                            request_id=f"eval-{run_id}-{query.id}-{policy}",
                        )
                        gateway_latency_ms = (time.perf_counter() - started_at) * 1000
                        response = gateway_answer.retrieval
                    else:
                        response = self.retrieval.retrieve(command)
                    retrieved = [hit.chunk_id for hit in response.hits]
                    metrics: dict[str, Any] = {
                        "recall_at_5": recall_at_k(retrieved, relevant, 5),
                        "recall_at_10": recall_at_k(retrieved, relevant, 10),
                        "mrr": reciprocal_rank(retrieved, relevant),
                        "ndcg_at_10": ndcg_at_k(retrieved, relevant, 10),
                        "latency_ms": response.timings_ms["total"],
                        "abstained": float(response.abstained),
                        "abstention_correct": float(response.abstained == (not query.answerable)),
                        "top_score": response.raw_top_score,
                        "abstention_threshold": response.abstention_threshold,
                    }
                    if gateway_answer is not None:
                        report_claims = (
                            list(gateway_answer.evidence_report.claims)
                            if gateway_answer.evidence_report
                            else []
                        )
                        unsupported_claims = [
                            claim.claim for claim in report_claims if claim.status != "supported"
                        ] or gateway_answer.unsupported_claims
                        contradiction_claim_count = sum(
                            claim.contradiction_score >= 0.5 for claim in report_claims
                        )
                        expected = " ".join(query.expected_facts)
                        fact_coverage = _token_recall(expected, gateway_answer.answer)
                        faithfulness = _lexical_faithfulness(
                            gateway_answer.answer,
                            [hit.text for hit in response.hits],
                        )
                        citation_validity = _citation_validity(gateway_answer)
                        abstained = _is_abstained(gateway_answer)
                        answered = not abstained
                        failure_type = _gateway_failure_type(
                            query=query,
                            answer=gateway_answer,
                            fact_coverage=fact_coverage,
                            citation_validity=citation_validity,
                        )
                        metrics.update(
                            {
                                "answer_faithfulness": faithfulness,
                                "answer_fact_coverage": fact_coverage,
                                "citation_validity": citation_validity,
                                "gateway_latency_ms": gateway_latency_ms or 0.0,
                                "gateway_failure_type": failure_type,
                            }
                        )
                        gateway_cases.append(
                            {
                                "id": query.id,
                                "query": query.query,
                                "language": query.language,
                                "tags": query.tags,
                                "answerable": query.answerable,
                                "policy": policy,
                                "selected_retrieval_policy": response.policy,
                                "selected_provider": gateway_answer.selected_provider,
                                "provider": gateway_answer.provider,
                                "answer": gateway_answer.answer,
                                "answered": answered,
                                "abstained": abstained,
                                "abstention_reason": gateway_answer.abstention_reason,
                                "grounded": gateway_answer.grounded,
                                "evidence_score": gateway_answer.evidence_score,
                                "answerability_score": gateway_answer.answerability_score,
                                "coverage_score": gateway_answer.coverage_score,
                                "support_score": gateway_answer.support_score,
                                "citation_validity": citation_validity,
                                "citation_count": len(gateway_answer.citations),
                                "citation_source_ids": [
                                    citation.chunk_id for citation in gateway_answer.citations
                                ],
                                "unsupported_claim_count": len(unsupported_claims),
                                "unsupported_claims": unsupported_claims,
                                "contradiction_claim_count": contradiction_claim_count,
                                "citation_repair_attempted": bool(
                                    gateway_answer.evidence_report
                                    and gateway_answer.evidence_report.repair_attempted
                                ),
                                "citation_repair_succeeded": bool(
                                    gateway_answer.evidence_report
                                    and gateway_answer.evidence_report.repair_succeeded
                                ),
                                "expected_facts": query.expected_facts,
                                "fact_coverage": fact_coverage,
                                "faithfulness": faithfulness,
                                "failure_type": failure_type,
                                "run_id": gateway_answer.run_id,
                                "trace_id": response.trace_id,
                                "latency_ms": gateway_latency_ms or response.timings_ms["total"],
                                "cost_estimated_usd": float(
                                    gateway_answer.cost.get("estimated_usd", 0.0)
                                ),
                            }
                        )
                    row["policies"][policy] = metrics
                    for metric, value in metrics.items():
                        if metric in {
                            "latency_ms",
                            "abstained",
                            "abstention_correct",
                            "answer_faithfulness",
                            "answer_fact_coverage",
                            "citation_validity",
                            "gateway_latency_ms",
                        } and isinstance(value, (float, int)):
                            aggregates[policy][metric].append(float(value))
                    if query.answerable:
                        for metric in ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
                            value = metrics[metric]
                            if value is not None:
                                aggregates[policy][metric].append(float(value))
                        ndcg = float(metrics["ndcg_at_10"] or 0)
                        language_scores[query.language][policy].append(ndcg)
                        for tag in query.tags:
                            tag_scores[tag][policy].append(ndcg)
                rows.append(row)

            summary: dict[str, dict[str, float]] = {}
            for policy in policies:
                values = aggregates[policy]
                summary[policy] = {
                    "recall_at_5": mean(values["recall_at_5"]),
                    "recall_at_10": mean(values["recall_at_10"]),
                    "mrr": mean(values["mrr"]),
                    "ndcg_at_10": mean(values["ndcg_at_10"]),
                    "latency_p50_ms": percentile(values["latency_ms"], 0.50),
                    "latency_p95_ms": percentile(values["latency_ms"], 0.95),
                    "estimated_serial_qps": 1000 / max(mean(values["latency_ms"]), 0.001),
                    "abstention_rate": mean(values["abstained"]),
                    "abstention_accuracy": mean(values["abstention_correct"]),
                }
                for optional_metric in (
                    "answer_faithfulness",
                    "answer_fact_coverage",
                    "citation_validity",
                    "gateway_latency_ms",
                ):
                    if values[optional_metric]:
                        summary[policy][optional_metric] = mean(values[optional_metric])

            gateway_evaluation: dict[str, Any] | None = (
                {
                    "summary": summarize_gateway_cases(gateway_cases),
                    "cases": gateway_cases,
                }
                if evaluate_answers
                else None
            )
            by_language = {
                language: {policy: mean(values) for policy, values in policy_values.items()}
                for language, policy_values in language_scores.items()
            }
            by_tag = {
                tag: {policy: mean(values) for policy, values in policy_values.items()}
                for tag, policy_values in tag_scores.items()
            }
            payload: dict[str, Any] = {
                "run_id": run_id,
                "knowledge_base": knowledge_base,
                "dataset": str(dataset_path),
                "summary": summary,
                "gateway_evaluation": gateway_evaluation,
                "by_language": by_language,
                "by_tag": by_tag,
                "queries": rows,
                "metadata": {
                    "query_count": len(queries),
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "policies": policies,
                    "evaluate_answers": evaluate_answers,
                    "dataset_sha256": dataset_digest,
                    "git_revision": _git_revision(),
                    "dense_model": self.settings.dense_model,
                    "sparse_model": self.settings.sparse_model,
                    "late_model": self.settings.late_model,
                    "pipeline_version": self.settings.pipeline_version,
                },
            }
            results_path = output_dir / "results.json"
            report_path = output_dir / "report.html"
            results_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            write_html_report(payload, report_path)
            mlflow.log_params(
                {
                    "knowledge_base": knowledge_base,
                    "dataset": str(dataset_path),
                    "dataset_sha256": dataset_digest,
                    "query_count": len(queries),
                }
            )
            mlflow.set_tags(
                {
                    "git_revision": payload["metadata"]["git_revision"],
                    "pipeline_version": self.settings.pipeline_version,
                    "benchmark_kind": "retrieval",
                }
            )
            if evaluate_answers:
                mlflow.set_tag("benchmark_kind", "gateway")
            for summary_policy, summary_values in summary.items():
                for metric, value in summary_values.items():
                    mlflow.log_metric(f"{summary_policy}.{metric}", value)
            if gateway_evaluation is not None:
                for metric, value in gateway_evaluation["summary"]["overall"].items():
                    mlflow.log_metric(f"gateway.{metric}", value)
                for policy, values in gateway_evaluation["summary"]["by_policy"].items():
                    for metric, value in values.items():
                        mlflow.log_metric(f"gateway.{policy}.{metric}", value)
            mlflow.log_artifacts(str(output_dir), artifact_path="benchmark")
            return {
                "run_id": run_id,
                "report_path": str(report_path),
                "results_path": str(results_path),
                "summary": summary,
                "gateway_summary": gateway_evaluation["summary"]
                if gateway_evaluation is not None
                else None,
                "by_language": payload["by_language"],
                "metadata": payload["metadata"],
            }
