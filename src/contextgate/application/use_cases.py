from __future__ import annotations

import builtins
import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from contextgate.application import dto
from contextgate.domain.evidence import abstention_reason, build_evidence_report, score_evidence
from contextgate.domain.gateway import AbstentionReason, AnswerResult, AnswerStatus
from contextgate.domain.models import CostRecord
from contextgate.domain.retrieval import RetrievalResult
from contextgate.observability.metrics import ESTIMATED_COST, GATE_DECISIONS, PROVIDER_LATENCY
from contextgate.ports.repositories import (
    BenchmarkJobRunner,
    CostLedger,
    IngestionJobRunner,
    JobQueue,
    JobRepository,
    ProviderRegistry,
    ResponseCache,
    RouterTrainingJobRunner,
    TraceStore,
    UnitOfWorkFactory,
)


@dataclass(slots=True)
class CreateJobResult:
    job: Any


def _bounded_correlation_id(value: str, *, max_length: int = 128) -> str:
    if len(value) <= max_length:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{value[: max_length - len(digest) - 1]}-{digest}"


def _dispatch_created_job(
    *,
    created: bool,
    kind: str,
    job: Any,
    job_queue: JobQueue,
    uow_factory: UnitOfWorkFactory,
) -> None:
    if not created:
        return
    job_queue.enqueue(kind, job.id)
    with uow_factory() as uow:
        uow.mark_job_enqueued(job.id)
        uow.commit()


class DispatchPendingJobs:
    def __init__(self, uow_factory: UnitOfWorkFactory, job_queue: JobQueue) -> None:
        self.uow_factory = uow_factory
        self.job_queue = job_queue

    def execute(self) -> int:
        with self.uow_factory() as uow:
            pending = uow.pending_job_dispatches()
        for kind, job_id in pending:
            self.job_queue.enqueue(kind, job_id)
            with self.uow_factory() as uow:
                uow.mark_job_enqueued(job_id)
                uow.commit()
        return len(pending)


class ManageKnowledgeBases:
    def __init__(self, repository) -> None:
        self.repository = repository

    def create(self, payload: dto.KnowledgeBaseCreate) -> Any:
        return self.repository.create(payload)

    def list_openai_models(self) -> list[dict[str, str]]:
        models = []
        for kb in self.repository.list():
            for policy in ("auto", "fast", "balanced", "accurate"):
                models.append(
                    {
                        "id": f"kb:{kb.slug}:{policy}",
                        "object": "model",
                        "owned_by": "contextgate",
                    }
                )
        return models

    def get(self, identifier: str) -> Any:
        return self.repository.get(identifier)

    def get_job(self, job_id: str) -> Any:
        return self.repository.get_job(job_id)

    def list(self) -> list[Any]:
        return self.repository.list()

    def list_documents(self, identifier: str) -> builtins.list[Any]:
        return self.repository.list_documents(identifier)


class ManagePolicies:
    def __init__(self, repository) -> None:
        self.repository = repository

    def create(self, payload: dto.PolicyCreateCommand) -> Any:
        return self.repository.create(payload)

    def list(self) -> list[Any]:
        return self.repository.list()

    def get(self, policy_id: str) -> Any:
        return self.repository.get(policy_id)

    def promote(self, policy_id: str) -> Any:
        return self.repository.promote(policy_id)


class ManageApiKeys:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def create(self, name: str, scopes: list[str]) -> tuple[Any, str]:
        return self.repository.create(name, scopes)

    def list(self) -> list[Any]:
        return self.repository.list()

    def rotate(self, key_id: str) -> tuple[Any, str]:
        return self.repository.rotate(key_id)

    def disable(self, key_id: str) -> Any:
        return self.repository.disable(key_id)


class ManageRouterVersions:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def list(self, knowledge_base: str) -> list[Any]:
        return self.repository.list(knowledge_base)


class IngestDocuments:
    def __init__(self, uow_factory: UnitOfWorkFactory, job_queue: JobQueue) -> None:
        self.uow_factory = uow_factory
        self.job_queue = job_queue

    def enqueue(
        self,
        *,
        knowledge_base: str,
        path: Path,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        with self.uow_factory() as uow:
            job, created = uow.create_job(
                kind="ingest",
                payload={
                    "knowledge_base": knowledge_base,
                    "path": str(path),
                    "metadata": metadata or {},
                },
                idempotency_key=idempotency_key,
            )
            uow.commit()
        _dispatch_created_job(
            created=created,
            kind="ingest",
            job=job,
            job_queue=self.job_queue,
            uow_factory=self.uow_factory,
        )
        return job


class SyncQdrantCollection:
    def __init__(self, uow_factory: UnitOfWorkFactory, job_queue: JobQueue) -> None:
        self.uow_factory = uow_factory
        self.job_queue = job_queue

    def enqueue(
        self,
        *,
        knowledge_base: str,
        request: dto.SyncQdrantCommand,
        idempotency_key: str | None = None,
    ) -> Any:
        with self.uow_factory() as uow:
            job, created = uow.create_job(
                kind="sync_qdrant",
                payload={
                    "knowledge_base": knowledge_base,
                    "source_collection": request.source_collection,
                },
                idempotency_key=idempotency_key,
            )
            uow.commit()
        _dispatch_created_job(
            created=created,
            kind="sync_qdrant",
            job=job,
            job_queue=self.job_queue,
            uow_factory=self.uow_factory,
        )
        return job


class RetrieveContext:
    def __init__(self, retrieval_gateway) -> None:
        self.retrieval_gateway = retrieval_gateway

    def execute(self, request: dto.RetrieveCommand) -> RetrievalResult:
        return self.retrieval_gateway.retrieve(request)


class AnswerWithEvidence:
    def __init__(
        self,
        graph_runtime,
        uow_factory: UnitOfWorkFactory,
        provider_registry: ProviderRegistry,
        cost_ledger: CostLedger,
        trace_store: TraceStore,
        cache: ResponseCache,
        policy_repository: Any | None = None,
        trace_content_mode: str = "full",
    ) -> None:
        self.graph_runtime = graph_runtime
        self.uow_factory = uow_factory
        self.provider_registry = provider_registry
        self.cost_ledger = cost_ledger
        self.trace_store = trace_store
        self.cache = cache
        self.policy_repository = policy_repository
        self.trace_content_mode = trace_content_mode

    def execute(
        self,
        request: dto.AnswerCommand,
        *,
        request_id: str | None = None,
        token_callback: Callable[[str], None] | None = None,
    ) -> AnswerResult:
        execution_started = perf_counter()
        correlation_id = _bounded_correlation_id(request_id or str(uuid4()))
        policy_snapshot = {
            "id": "builtin-request-policy",
            "status": "active",
            "retrieval_policy": request.policy,
            "provider_policy": request.llm_provider,
            "latency_budget_ms": request.latency_budget_ms,
            "cost_budget_usd": request.cost_budget_usd,
            "max_context_tokens": request.max_context_tokens,
        }
        if request.gateway_policy_id:
            if self.policy_repository is None:
                raise RuntimeError("Gateway policy repository is not configured")
            policy = self.policy_repository.resolve_active(request.gateway_policy_id)
            request = replace(
                request,
                policy=policy.retrieval_policy,
                llm_provider=policy.provider_policy,
                latency_budget_ms=policy.latency_budget_ms,
                cost_budget_usd=policy.cost_budget_usd,
            )
            policy_snapshot = {
                "id": policy.id,
                "name": policy.name,
                "status": policy.status,
                "retrieval_policy": policy.retrieval_policy,
                "provider_policy": policy.provider_policy,
                "latency_budget_ms": policy.latency_budget_ms,
                "cost_budget_usd": policy.cost_budget_usd,
                "max_context_tokens": request.max_context_tokens,
                "promoted_at": policy.promoted_at.isoformat() if policy.promoted_at else None,
            }
        run_id = str(uuid4())
        start_run = getattr(self.trace_store, "start_run", None)
        if callable(start_run):
            start_run(
                run_id,
                correlation_id=correlation_id,
                knowledge_base=request.knowledge_base,
                query=self._trace_text(request.query),
            )
        runtime_request = replace(
            request,
            request_id=run_id,
            deadline_monotonic=execution_started + request.latency_budget_ms / 1000,
        )
        try:
            try:
                response = self.graph_runtime.answer(
                    runtime_request,
                    token_callback=token_callback,
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                response = self.graph_runtime.answer(runtime_request)
        except Exception as exc:
            self._record_failure(run_id, request, exc)
            raise
        elapsed_ms = (perf_counter() - execution_started) * 1000
        contexts = [hit.text for hit in response.retrieval.hits]
        evidence = score_evidence(
            query=request.query,
            answer=response.answer,
            contexts=contexts,
            abstained=response.retrieval.abstained,
        )
        report = response.evidence_report
        if response.status == AnswerStatus.ANSWERED and report is None:
            report = build_evidence_report(
                answer=response.answer,
                citations=response.citations,
                hits=response.retrieval.hits,
                require_citations=True,
            )
        reason = (
            response.abstention_reason
            or (report.reason if report else None)
            or abstention_reason(
                evidence,
                retrieval_empty=response.retrieval.abstained or not response.retrieval.hits,
            )
        )
        actual_usd = response.cost.get("actual_usd")
        if (
            request.cost_budget_usd is not None
            and actual_usd is not None
            and float(actual_usd) > request.cost_budget_usd
        ):
            reason = AbstentionReason.BUDGET_EXCEEDED
        if elapsed_ms > request.latency_budget_ms:
            reason = AbstentionReason.LATENCY_BUDGET_EXCEEDED
        response = replace(
            response,
            run_id=run_id,
            evidence_score=report.score if report else evidence.score,
            answerability_score=evidence.answerability_score,
            coverage_score=evidence.coverage_score,
            support_score=evidence.support_score,
            unsupported_claims=list(evidence.unsupported_claims),
            rejected_claims=list(evidence.rejected_claims),
            evidence_report=report,
            policy_snapshot=policy_snapshot,
        )
        if reason is not None and response.status != AnswerStatus.BLOCKED:
            response = self._force_abstention(response, reason)
        provider = self._selected_provider(runtime_request, response)
        response = replace(response, selected_provider=provider)
        self._append_trace_events(
            run_id,
            request,
            response,
            correlation_id=correlation_id,
        )
        actual_cost = response.cost.get("actual_usd")
        estimated_cost = response.cost.get("estimated_usd", 0.0)
        billed_provider = str(response.cost.get("provider") or provider)
        self.cost_ledger.record(
            CostRecord(
                request_id=correlation_id,
                run_id=run_id,
                provider=billed_provider,
                model=billed_provider,
                input_tokens=int(response.cost.get("input_tokens", 0)),
                output_tokens=int(response.cost.get("output_tokens", 0)),
                embedding_units=len(contexts),
                rerank_units=0,
                estimated_cost_usd=float(
                    actual_cost if actual_cost is not None else estimated_cost or 0.0
                ),
            )
        )
        GATE_DECISIONS.labels(
            status=response.status,
            reason=response.abstention_reason or "none",
            retrieval_policy=response.retrieval.policy,
            provider=provider,
        ).inc()
        PROVIDER_LATENCY.labels(provider=billed_provider).observe(
            float(response.cost.get("latency_ms", 0.0)) / 1000
        )
        ESTIMATED_COST.labels(provider=billed_provider).observe(
            float(actual_cost if actual_cost is not None else estimated_cost or 0.0)
        )
        return response

    def _trace_text(self, value: str) -> str:
        if self.trace_content_mode == "full":
            return value
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"<redacted:sha256:{digest}>"

    def _record_failure(self, run_id: str, request: dto.AnswerCommand, exc: Exception) -> None:
        self.trace_store.append_event(
            run_id,
            "final",
            {
                "trace_id": run_id,
                "knowledge_base": request.knowledge_base,
                "query": self._trace_text(request.query),
                "provider": "",
                "retrieval_policy": request.policy,
                "status": "failed",
                "abstained": False,
                "error": {"type": exc.__class__.__name__},
            },
        )

    def _selected_provider(self, request: dto.AnswerCommand, response: AnswerResult) -> str:
        if response.status == AnswerStatus.BLOCKED:
            return "blocked"
        if (
            response.provider == "abstention"
            or response.retrieval.abstained
            or response.abstention_reason is not None
        ):
            return "abstention"
        if response.provider == "extractive":
            return "extractive"
        return response.provider

    def _force_abstention(
        self,
        response: AnswerResult,
        reason: AbstentionReason,
    ) -> AnswerResult:
        return replace(
            response,
            answer="",
            citations=[],
            provider="abstention",
            selected_provider="abstention",
            grounded=False,
            status=AnswerStatus.ABSTAINED,
            abstention_reason=reason,
        )

    def _append_trace_events(
        self,
        run_id: str,
        request: dto.AnswerCommand,
        response: AnswerResult,
        *,
        correlation_id: str,
    ) -> None:
        retrieval = response.retrieval
        events: list[tuple[str, dict[str, Any]]] = []

        def emit(event_type: str, payload: dict[str, Any]) -> None:
            events.append((event_type, payload))

        emit(
            "query_analyzed",
            {
                "query": self._trace_text(request.query),
                "language": retrieval.features.get("language"),
                "query_token_count": retrieval.features.get("query_token_count"),
                "latency_budget_ms": request.latency_budget_ms,
                "cost_budget_usd": request.cost_budget_usd,
            },
        )
        emit(
            "retrieval_started",
            {
                "requested_policy": retrieval.route.requested_policy,
                "selected_policy": retrieval.policy,
                "route_reason": retrieval.route.reason,
            },
        )
        emit(
            "risk_checked",
            asdict(response.risk_report)
            if response.risk_report
            else {"score": 0.0, "blocked": False},
        )
        for hit in retrieval.hits:
            emit(
                "retrieval_hit",
                {
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.document_id,
                    "source": hit.source,
                    "rank": hit.rank,
                    "score": hit.score,
                },
            )
        emit(
            "evidence_scored",
            {
                "evidence_score": response.evidence_score,
                "answerability_score": response.answerability_score,
                "coverage_score": response.coverage_score,
                "support_score": response.support_score,
                "unsupported_claims": response.unsupported_claims,
                "rejected_claims": response.rejected_claims,
                "evidence_report": asdict(response.evidence_report)
                if response.evidence_report
                else None,
                "generation_allowed": response.abstention_reason is None
                and response.status == AnswerStatus.ANSWERED,
            },
        )
        emit(
            "provider_selected",
            {
                "provider": response.selected_provider,
                "generation_provider": response.cost.get("provider", response.provider),
                "model": response.cost.get("provider", response.provider),
                "allowed_providers": request.allowed_providers,
            },
        )
        if response.answer:
            tokens = response.answer.split()
            for offset in range(0, len(tokens), 20):
                emit(
                    "token_delta",
                    {"text": " ".join(tokens[offset : offset + 20]) + " "},
                )
        emit(
            "citation_verified",
            {
                "grounded": response.grounded,
                "citations": [asdict(citation) for citation in response.citations],
            },
        )
        emit(
            "final",
            {
                "trace_id": retrieval.trace_id,
                "knowledge_base": request.knowledge_base,
                "query": self._trace_text(request.query),
                "provider": response.selected_provider,
                "retrieval_policy": retrieval.policy,
                "evidence_score": response.evidence_score,
                "answerability_score": response.answerability_score,
                "coverage_score": response.coverage_score,
                "support_score": response.support_score,
                "abstained": response.abstention_reason is not None
                or response.provider == "abstention"
                or retrieval.abstained,
                "abstention_reason": response.abstention_reason,
                "status": response.status,
                "correlation_id": correlation_id,
                "policy_snapshot": response.policy_snapshot,
                "corpus_version": retrieval.features.get("corpus_version", 0),
                "verifier": response.evidence_report.verifier if response.evidence_report else None,
                "verifier_version": response.evidence_report.verifier_version
                if response.evidence_report
                else None,
                "cost": response.cost,
            },
        )
        append_events = getattr(self.trace_store, "append_events", None)
        if callable(append_events):
            append_events(run_id, events)
            return
        for event_type, payload in events:
            self.trace_store.append_event(run_id, event_type, payload)

    def stream_events(
        self, request: dto.AnswerCommand, *, request_id: str | None = None
    ) -> list[dict[str, Any]]:
        response = self.execute(request, request_id=request_id)
        return [
            {"event": "query_analyzed", "data": {"query": request.query}},
            {"event": "retrieval_started", "data": {"policy": response.retrieval.policy}},
            {
                "event": "evidence_scored",
                "data": {
                    "evidence_score": response.evidence_score,
                    "answerability_score": response.answerability_score,
                },
            },
            {"event": "provider_selected", "data": {"provider": response.selected_provider}},
            {"event": "citation_verified", "data": {"grounded": response.grounded}},
            {"event": "final", "data": asdict(response)},
        ]


class RunBenchmark:
    def __init__(self, uow_factory: UnitOfWorkFactory, job_queue: JobQueue) -> None:
        self.uow_factory = uow_factory
        self.job_queue = job_queue

    def enqueue(self, request: dto.BenchmarkCommand, idempotency_key: str | None = None) -> Any:
        with self.uow_factory() as uow:
            job, created = uow.create_job(
                kind="benchmark",
                payload=asdict(request),
                idempotency_key=idempotency_key,
            )
            uow.commit()
        _dispatch_created_job(
            created=created,
            kind="benchmark",
            job=job,
            job_queue=self.job_queue,
            uow_factory=self.uow_factory,
        )
        return job


class TrainRouter:
    def __init__(self, uow_factory: UnitOfWorkFactory, job_queue: JobQueue) -> None:
        self.uow_factory = uow_factory
        self.job_queue = job_queue

    def enqueue(self, request: dto.RouterTrainCommand, idempotency_key: str | None = None) -> Any:
        with self.uow_factory() as uow:
            job, created = uow.create_job(
                kind="router_train",
                payload=asdict(request),
                idempotency_key=idempotency_key,
            )
            uow.commit()
        _dispatch_created_job(
            created=created,
            kind="router_train",
            job=job,
            job_queue=self.job_queue,
            uow_factory=self.uow_factory,
        )
        return job


class ExecuteIngestJob:
    def __init__(self, jobs: JobRepository, runner: IngestionJobRunner) -> None:
        self.jobs = jobs
        self.runner = runner

    def execute(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.start(job_id)
        try:
            result = self.runner.ingest(job_id, job.payload)
            if result.get("outcome") == "failed":
                self.jobs.fail(
                    job_id,
                    "All documents failed ingestion",
                    {"type": "IngestionFailed", "failures": result.get("failures", [])},
                )
            else:
                self.jobs.succeed(job_id, result)
            return result
        except Exception as exc:
            self.jobs.fail(
                job_id,
                str(exc),
                {"type": exc.__class__.__name__, "message": str(exc)},
            )
            raise


class ExecuteSyncQdrantJob:
    def __init__(self, jobs: JobRepository, runner: IngestionJobRunner) -> None:
        self.jobs = jobs
        self.runner = runner

    def execute(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.start(job_id)
        try:
            result = self.runner.sync_qdrant(job_id, job.payload)
            self.jobs.succeed(job_id, result)
            return result
        except Exception as exc:
            self.jobs.fail(
                job_id,
                str(exc),
                {"type": exc.__class__.__name__, "message": str(exc)},
            )
            raise


class ExecuteBenchmarkJob:
    def __init__(self, jobs: JobRepository, runner: BenchmarkJobRunner) -> None:
        self.jobs = jobs
        self.runner = runner

    def execute(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.start(job_id)
        try:
            result = self.runner.run(job.payload)
            self.jobs.succeed(job_id, result)
            return result
        except Exception as exc:
            self.jobs.fail(
                job_id,
                str(exc),
                {"type": exc.__class__.__name__, "message": str(exc)},
            )
            raise


class ExecuteTrainRouterJob:
    def __init__(self, jobs: JobRepository, runner: RouterTrainingJobRunner) -> None:
        self.jobs = jobs
        self.runner = runner

    def execute(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.start(job_id)
        try:
            result = self.runner.train(job.payload)
            self.jobs.succeed(job_id, result)
            return result
        except Exception as exc:
            self.jobs.fail(
                job_id,
                str(exc),
                {"type": exc.__class__.__name__, "message": str(exc)},
            )
            raise


class CancelJob:
    def __init__(self, jobs: JobRepository, job_queue: JobQueue) -> None:
        self.jobs = jobs
        self.job_queue = job_queue

    def execute(self, job_id: str) -> Any:
        job = self.jobs.cancel(job_id)
        self.job_queue.cancel(job_id)
        return job


class PromotePolicy:
    def __init__(self, router_manager, uow_factory: UnitOfWorkFactory) -> None:
        self.router_manager = router_manager
        self.uow_factory = uow_factory

    def execute(self, request: dto.RouterPromoteCommand) -> dict[str, str]:
        target = self.router_manager.promote(request.run_id, request.knowledge_base)
        with self.uow_factory() as uow:
            uow.promote_router_version(request.run_id, request.knowledge_base)
            uow.commit()
        return {"status": "promoted", "path": str(target)}


class InspectTrace:
    def __init__(self, trace_store: TraceStore, cost_ledger: CostLedger) -> None:
        self.trace_store = trace_store
        self.cost_ledger = cost_ledger

    def run(self, run_id: str) -> dict[str, Any]:
        return self.trace_store.get_trace(run_id)

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return self.trace_store.list_events(run_id)

    def cost(self, run_id: str) -> dict[str, Any]:
        records = self.cost_ledger.list_for_run(run_id)
        return {
            "run_id": run_id,
            "records": [asdict(record) for record in records],
            "estimated_usd": sum(record.estimated_cost_usd for record in records),
        }

    def purge(self, days: int) -> int:
        return self.trace_store.purge_older_than(days)
