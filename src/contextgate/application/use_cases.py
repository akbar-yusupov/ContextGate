from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from contextgate.application import dto
from contextgate.domain.evidence import abstention_reason, score_evidence
from contextgate.domain.gateway import AbstentionReason, AnswerResult
from contextgate.domain.models import CostRecord
from contextgate.domain.retrieval import RetrievalResult
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

    def get_job(self, job_id: str) -> Any:
        return self.repository.get_job(job_id)


class ManagePolicies:
    def __init__(self, repository) -> None:
        self.repository = repository

    def create(self, payload: dto.PolicyCreateCommand) -> Any:
        return self.repository.create(payload)

    def get(self, policy_id: str) -> Any:
        return self.repository.get(policy_id)

    def promote(self, policy_id: str) -> Any:
        return self.repository.promote(policy_id)


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
            job = uow.create_job(
                kind="ingest",
                payload={
                    "knowledge_base": knowledge_base,
                    "path": str(path),
                    "metadata": metadata or {},
                },
                idempotency_key=idempotency_key,
            )
            uow.commit()
        self.job_queue.enqueue("ingest", job.id)
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
            job = uow.create_job(
                kind="sync_qdrant",
                payload={
                    "knowledge_base": knowledge_base,
                    "source_collection": request.source_collection,
                },
                idempotency_key=idempotency_key,
            )
            uow.commit()
        self.job_queue.enqueue("sync_qdrant", job.id)
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
    ) -> None:
        self.graph_runtime = graph_runtime
        self.uow_factory = uow_factory
        self.provider_registry = provider_registry
        self.cost_ledger = cost_ledger
        self.trace_store = trace_store
        self.cache = cache

    def execute(
        self,
        request: dto.AnswerCommand,
        *,
        request_id: str | None = None,
    ) -> AnswerResult:
        request_id = request_id or str(uuid4())
        cache_key = (
            f"{request.knowledge_base}:{request.policy}:{request.query}:{request.llm_provider}"
        )
        use_cache = not request.debug
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        runtime_request = replace(request, request_id=request_id)
        response = self.graph_runtime.answer(runtime_request)
        provider = self._selected_provider(request, response)
        contexts = [hit.text for hit in response.retrieval.hits]
        evidence = score_evidence(
            query=request.query,
            answer=response.answer,
            contexts=contexts,
            abstained=response.retrieval.abstained,
        )
        reason = response.abstention_reason or abstention_reason(
            evidence,
            retrieval_empty=response.retrieval.abstained or not response.retrieval.hits,
        )
        response = replace(
            response,
            run_id=request_id,
            evidence_score=evidence.score,
            answerability_score=evidence.answerability_score,
            coverage_score=evidence.coverage_score,
            support_score=evidence.support_score,
            unsupported_claims=list(evidence.unsupported_claims),
            rejected_claims=list(evidence.rejected_claims),
            cost={"estimated_usd": 0.0},
        )
        if reason is not None:
            response = self._force_abstention(response, reason)
        provider = self._selected_provider(request, response)
        response = replace(response, selected_provider=provider)
        self._append_trace_events(request_id, request, response)
        self.cost_ledger.record(
            CostRecord(
                request_id=request_id,
                run_id=request_id,
                provider=provider,
                model=provider,
                input_tokens=len(request.query.split()),
                output_tokens=len(response.answer.split()),
                embedding_units=len(contexts),
                rerank_units=0,
                estimated_cost_usd=0.0,
            )
        )
        if use_cache:
            self.cache.set(cache_key, response)
        return response

    def _selected_provider(self, request: dto.AnswerCommand, response: AnswerResult) -> str:
        if (
            response.provider == "abstention"
            or response.retrieval.abstained
            or response.abstention_reason is not None
        ):
            return "abstention"
        if response.provider == "extractive":
            return "extractive"
        return self.provider_registry.choose(
            cost_budget_usd=request.cost_budget_usd,
            latency_budget_ms=request.latency_budget_ms,
            allowed_providers=request.allowed_providers,
            requested_provider=request.llm_provider,
        )

    def _force_abstention(
        self,
        response: AnswerResult,
        reason: AbstentionReason,
    ) -> AnswerResult:
        return replace(
            response,
            answer=(
                "I could not answer from grounded evidence in the knowledge base. "
                f"Abstention reason: {reason.value}."
            ),
            citations=[],
            provider="abstention",
            selected_provider="abstention",
            grounded=False,
            abstention_reason=reason,
        )

    def _append_trace_events(
        self,
        run_id: str,
        request: dto.AnswerCommand,
        response: AnswerResult,
    ) -> None:
        retrieval = response.retrieval
        self.trace_store.append_event(
            run_id,
            "query_analyzed",
            {
                "query": request.query,
                "language": retrieval.features.get("language"),
                "query_token_count": retrieval.features.get("query_token_count"),
                "latency_budget_ms": request.latency_budget_ms,
                "cost_budget_usd": request.cost_budget_usd,
            },
        )
        self.trace_store.append_event(
            run_id,
            "retrieval_started",
            {
                "requested_policy": retrieval.route.requested_policy,
                "selected_policy": retrieval.policy,
                "route_reason": retrieval.route.reason,
            },
        )
        for hit in retrieval.hits:
            self.trace_store.append_event(
                run_id,
                "retrieval_hit",
                {
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.document_id,
                    "source": hit.source,
                    "rank": hit.rank,
                    "score": hit.score,
                },
            )
        self.trace_store.append_event(
            run_id,
            "evidence_scored",
            {
                "evidence_score": response.evidence_score,
                "answerability_score": response.answerability_score,
                "coverage_score": response.coverage_score,
                "support_score": response.support_score,
                "unsupported_claims": response.unsupported_claims,
                "rejected_claims": response.rejected_claims,
                "generation_allowed": response.abstention_reason is None
                and response.provider != "abstention",
            },
        )
        self.trace_store.append_event(
            run_id,
            "provider_selected",
            {
                "provider": response.selected_provider,
                "model": response.provider,
                "allowed_providers": request.allowed_providers,
            },
        )
        if response.answer:
            self.trace_store.append_event(run_id, "token_delta", {"text": response.answer})
        self.trace_store.append_event(
            run_id,
            "citation_verified",
            {
                "grounded": response.grounded,
                "citations": [asdict(citation) for citation in response.citations],
            },
        )
        self.trace_store.append_event(
            run_id,
            "final",
            {
                "trace_id": retrieval.trace_id,
                "knowledge_base": request.knowledge_base,
                "query": request.query,
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
                "cost": response.cost,
            },
        )

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
            job = uow.create_job(
                kind="benchmark",
                payload=asdict(request),
                idempotency_key=idempotency_key,
            )
            uow.commit()
        self.job_queue.enqueue("benchmark", job.id)
        return job


class TrainRouter:
    def __init__(self, uow_factory: UnitOfWorkFactory, job_queue: JobQueue) -> None:
        self.uow_factory = uow_factory
        self.job_queue = job_queue

    def enqueue(self, request: dto.RouterTrainCommand, idempotency_key: str | None = None) -> Any:
        with self.uow_factory() as uow:
            job = uow.create_job(
                kind="router_train",
                payload=asdict(request),
                idempotency_key=idempotency_key,
            )
            uow.commit()
        self.job_queue.enqueue("router_train", job.id)
        return job


class ExecuteIngestJob:
    def __init__(self, jobs: JobRepository, runner: IngestionJobRunner) -> None:
        self.jobs = jobs
        self.runner = runner

    def execute(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.start(job_id)
        try:
            result = self.runner.ingest(job_id, job.payload)
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


class PromotePolicy:
    def __init__(self, router_manager, uow_factory: UnitOfWorkFactory) -> None:
        self.router_manager = router_manager
        self.uow_factory = uow_factory

    def execute(self, request: dto.RouterPromoteCommand) -> dict[str, str]:
        target = self.router_manager.promote(request.run_id, request.knowledge_base)
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
