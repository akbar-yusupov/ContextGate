from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mlflow
from sqlalchemy import text

from contextgate.adapters.celery.app import celery_app
from contextgate.adapters.celery.job_runners import (
    BenchmarkServiceJobRunner,
    IngestionServiceJobRunner,
    RouterTrainingServiceJobRunner,
)
from contextgate.adapters.celery.queue import CeleryJobQueue
from contextgate.adapters.langgraph.checkpointing import (
    CheckpointerResource,
    create_postgres_checkpointer,
)
from contextgate.adapters.langgraph.runtime import GatewayGraph
from contextgate.adapters.litellm.providers import ProviderRegistry
from contextgate.adapters.local.guardrails import SemanticClaimVerifier
from contextgate.adapters.local.ingestion_service import IngestionService
from contextgate.adapters.mlflow.evaluation_store import BenchmarkService
from contextgate.adapters.mlflow.router_registry import RouterManager
from contextgate.adapters.qdrant.vector_index import get_vector_store
from contextgate.adapters.sqlalchemy import SessionLocal, init_db
from contextgate.adapters.sqlalchemy.ledger import (
    InMemoryResponseCache,
    SqlAlchemyCostLedger,
    SqlAlchemyTraceStore,
)
from contextgate.adapters.sqlalchemy.repositories import (
    SqlAlchemyApiKeyRepository,
    SqlAlchemyJobRepository,
    SqlAlchemyKnowledgeBaseRepository,
    SqlAlchemyPolicyRepository,
    SqlAlchemyRouterVersionRepository,
)
from contextgate.adapters.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWorkFactory
from contextgate.application.retrieval import RetrievalService
from contextgate.application.use_cases import (
    AnswerWithEvidence,
    CancelJob,
    DispatchPendingJobs,
    ExecuteBenchmarkJob,
    ExecuteIngestJob,
    ExecuteSyncQdrantJob,
    ExecuteTrainRouterJob,
    IngestDocuments,
    InspectTrace,
    ManageApiKeys,
    ManageKnowledgeBases,
    ManagePolicies,
    ManageRouterVersions,
    PromotePolicy,
    RetrieveContext,
    RunBenchmark,
    SyncQdrantCollection,
    TrainRouter,
)
from contextgate.apps.api.dependencies import ensure_bootstrap_api_key
from contextgate.config import Settings, get_policies, get_settings


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    knowledge_bases: ManageKnowledgeBases
    policies: ManagePolicies
    router_versions: ManageRouterVersions
    api_keys: ManageApiKeys
    ingest_documents: IngestDocuments
    sync_qdrant: SyncQdrantCollection
    retrieve_context: RetrieveContext
    answer_with_evidence: AnswerWithEvidence
    run_benchmark: RunBenchmark
    train_router: TrainRouter
    cancel_job: CancelJob
    dispatch_pending_jobs: DispatchPendingJobs
    execute_ingest_job: ExecuteIngestJob
    execute_sync_qdrant_job: ExecuteSyncQdrantJob
    execute_benchmark_job: ExecuteBenchmarkJob
    execute_train_router_job: ExecuteTrainRouterJob
    promote_policy: PromotePolicy
    inspect_trace: InspectTrace
    provider_registry: ProviderRegistry
    ingestion_service: IngestionService
    benchmark_service: BenchmarkService
    router_manager: RouterManager
    langgraph_checkpointer: CheckpointerResource | None = None

    def startup(self) -> None:
        self.settings.validate_runtime_security()
        init_db()
        mlflow.set_tracking_uri(self.settings.resolved_mlflow_tracking_uri)
        with SessionLocal() as session:
            ensure_bootstrap_api_key(session, self.settings)
        self.inspect_trace.purge(self.settings.trace_retention_days)
        self.dispatch_pending_jobs.execute()

    def shutdown(self) -> None:
        if get_vector_store.cache_info().currsize:
            get_vector_store().close()
        if self.langgraph_checkpointer is not None:
            self.langgraph_checkpointer.close()

    def readiness(self) -> dict[str, str]:
        checks: dict[str, str] = {}
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
            checks["database"] = "ok"
        get_vector_store().client.get_collections()
        checks["qdrant"] = "ok"
        return checks


@lru_cache
def get_container() -> AppContainer:
    settings = get_settings()

    uow_factory = SqlAlchemyUnitOfWorkFactory(SessionLocal)
    job_queue = CeleryJobQueue(celery_app)
    provider_registry = ProviderRegistry(
        settings.llm_model,
        input_cost_per_1m_tokens=settings.llm_input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=settings.llm_output_cost_per_1m_tokens,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    cost_ledger = SqlAlchemyCostLedger(SessionLocal)
    trace_store = SqlAlchemyTraceStore(SessionLocal)
    cache = InMemoryResponseCache()
    ingestion_service = IngestionService(settings=settings)
    knowledge_base_repository = SqlAlchemyKnowledgeBaseRepository(SessionLocal)
    job_repository = SqlAlchemyJobRepository(SessionLocal)
    policy_repository = SqlAlchemyPolicyRepository(SessionLocal)
    api_key_repository = SqlAlchemyApiKeyRepository(SessionLocal)
    router_version_repository = SqlAlchemyRouterVersionRepository(SessionLocal)
    router_manager = RouterManager(settings, router_version_repository.active_artifact)
    ingestion_runner = IngestionServiceJobRunner(ingestion_service, SessionLocal)
    router_training_runner = RouterTrainingServiceJobRunner(router_manager, SessionLocal)
    retrieval_service = RetrievalService(
        vector_index=get_vector_store(),
        policies=get_policies(),
        router=router_manager,
        knowledge_bases=knowledge_base_repository,
    )
    checkpointer = (
        create_postgres_checkpointer(settings.resolved_database_url)
        if settings.graph_checkpointing_enabled
        else None
    )
    graph = GatewayGraph(
        retrieval=retrieval_service,
        claim_verifier=SemanticClaimVerifier(get_vector_store().embedder),
        provider_registry=provider_registry,
        checkpointer=checkpointer.saver if checkpointer else None,
    )
    answer_with_evidence = AnswerWithEvidence(
        graph,
        uow_factory,
        provider_registry,
        cost_ledger,
        trace_store,
        cache,
        policy_repository,
        settings.trace_content_mode,
    )
    benchmark_service = BenchmarkService(
        retrieval=retrieval_service,
        settings=settings,
        answer_gateway=answer_with_evidence,
    )
    benchmark_runner = BenchmarkServiceJobRunner(benchmark_service, SessionLocal)
    return AppContainer(
        settings=settings,
        knowledge_bases=ManageKnowledgeBases(knowledge_base_repository),
        policies=ManagePolicies(policy_repository),
        router_versions=ManageRouterVersions(router_version_repository),
        api_keys=ManageApiKeys(api_key_repository),
        ingest_documents=IngestDocuments(uow_factory, job_queue),
        sync_qdrant=SyncQdrantCollection(uow_factory, job_queue),
        retrieve_context=RetrieveContext(retrieval_service),
        answer_with_evidence=answer_with_evidence,
        run_benchmark=RunBenchmark(uow_factory, job_queue),
        train_router=TrainRouter(uow_factory, job_queue),
        cancel_job=CancelJob(job_repository, job_queue),
        dispatch_pending_jobs=DispatchPendingJobs(uow_factory, job_queue),
        execute_ingest_job=ExecuteIngestJob(job_repository, ingestion_runner),
        execute_sync_qdrant_job=ExecuteSyncQdrantJob(job_repository, ingestion_runner),
        execute_benchmark_job=ExecuteBenchmarkJob(job_repository, benchmark_runner),
        execute_train_router_job=ExecuteTrainRouterJob(job_repository, router_training_runner),
        promote_policy=PromotePolicy(router_manager, uow_factory),
        inspect_trace=InspectTrace(trace_store, cost_ledger),
        provider_registry=provider_registry,
        ingestion_service=ingestion_service,
        benchmark_service=benchmark_service,
        router_manager=router_manager,
        langgraph_checkpointer=checkpointer,
    )
