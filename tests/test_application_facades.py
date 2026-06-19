from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from contextgate.application import dto
from contextgate.application.use_cases import (
    CancelJob,
    ExecuteIngestJob,
    ExecuteSyncQdrantJob,
    ExecuteTrainRouterJob,
    InspectTrace,
    ManageApiKeys,
    ManagePolicies,
    ManageRouterVersions,
    PromotePolicy,
    RetrieveContext,
    RunBenchmark,
    SyncQdrantCollection,
    TrainRouter,
)
from contextgate.ports.repositories import JobQueue, JobRepository, UnitOfWorkFactory


class RecordingRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _record(self, name: str, *args: Any) -> str:
        self.calls.append((name, args))
        return name

    def create(self, *args: Any) -> Any:
        return self._record("create", *args)

    def list(self, *args: Any) -> Any:
        return [self._record("list", *args)]

    def get(self, *args: Any) -> Any:
        return self._record("get", *args)

    def promote(self, *args: Any) -> Any:
        return self._record("promote", *args)

    def rotate(self, *args: Any) -> Any:
        return self._record("rotate", *args), "secret"

    def disable(self, *args: Any) -> Any:
        return self._record("disable", *args)


class RecordingUnitOfWork:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.enqueued: list[str] = []
        self.promoted: list[tuple[str, str]] = []
        self.commits = 0

    def __enter__(self) -> RecordingUnitOfWork:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def create_job(self, **kwargs: Any) -> tuple[Any, bool]:
        self.created.append(kwargs)
        return SimpleNamespace(id=f"job-{len(self.created)}", payload=kwargs["payload"]), True

    def mark_job_enqueued(self, job_id: str) -> None:
        self.enqueued.append(job_id)

    def promote_router_version(self, run_id: str, knowledge_base: str) -> None:
        self.promoted.append((run_id, knowledge_base))

    def commit(self) -> None:
        self.commits += 1


class RecordingQueue:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def enqueue(self, kind: str, job_id: str) -> None:
        self.sent.append((kind, job_id))


def test_management_facades_delegate_without_changing_contracts() -> None:
    repository = RecordingRepository()
    policy = ManagePolicies(repository)
    keys = ManageApiKeys(repository)
    routers = ManageRouterVersions(repository)
    command = dto.PolicyCreateCommand(name="strict")

    assert policy.create(command) == "create"
    assert policy.list() == ["list"]
    assert policy.get("policy-1") == "get"
    assert policy.promote("policy-1") == "promote"
    assert keys.create("client", ["read"]) == "create"
    assert keys.list() == ["list"]
    assert keys.rotate("key-1") == ("rotate", "secret")
    assert keys.disable("key-1") == "disable"
    assert routers.list("demo") == ["list"]


def test_retrieval_and_background_enqueue_facades_preserve_payloads() -> None:
    class Retrieval:
        def retrieve(self, request: dto.RetrieveCommand) -> Any:
            return request

    retrieve_command = dto.RetrieveCommand(knowledge_base="demo", query="question")
    assert RetrieveContext(Retrieval()).execute(retrieve_command) is retrieve_command

    uow = RecordingUnitOfWork()

    def factory() -> RecordingUnitOfWork:
        return uow

    queue = RecordingQueue()

    uow_factory = cast(UnitOfWorkFactory, factory)
    job_queue = cast(JobQueue, queue)
    synced = SyncQdrantCollection(uow_factory, job_queue).enqueue(
        knowledge_base="demo",
        request=dto.SyncQdrantCommand(source_collection="legacy"),
        idempotency_key="sync-1",
    )
    benchmark = RunBenchmark(uow_factory, job_queue).enqueue(
        dto.BenchmarkCommand(knowledge_base="demo", dataset_path=str(Path("cases.jsonl"))),
        idempotency_key="bench-1",
    )
    trained = TrainRouter(uow_factory, job_queue).enqueue(
        dto.RouterTrainCommand(benchmark_run_id="bench-run", knowledge_base="demo"),
        idempotency_key="train-1",
    )

    assert [item["kind"] for item in uow.created] == [
        "sync_qdrant",
        "benchmark",
        "router_train",
    ]
    assert queue.sent == [
        ("sync_qdrant", synced.id),
        ("benchmark", benchmark.id),
        ("router_train", trained.id),
    ]
    assert uow.enqueued == [synced.id, benchmark.id, trained.id]


def test_job_execution_promotion_cancellation_and_trace_facades() -> None:
    class Jobs:
        def __init__(self) -> None:
            self.succeeded: list[tuple[str, dict[str, Any]]] = []

        def start(self, job_id: str) -> Any:
            return SimpleNamespace(id=job_id, payload={"job_id": job_id})

        def succeed(self, job_id: str, result: dict[str, Any]) -> None:
            self.succeeded.append((job_id, result))

        def fail(self, *_: Any) -> None:
            raise AssertionError("successful execution must not fail the job")

        def cancel(self, job_id: str) -> Any:
            return SimpleNamespace(id=job_id, status="cancelled")

    class Runners:
        def ingest(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
            return {"outcome": "succeeded", "job_id": job_id, **payload}

        def sync_qdrant(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
            return {"outcome": "succeeded", "job_id": job_id, **payload}

        def train(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"run_id": "router-1", **payload}

    class Queue:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def cancel(self, job_id: str) -> None:
            self.cancelled.append(job_id)

    jobs = Jobs()
    runners = Runners()
    job_repository = cast(JobRepository, jobs)
    assert ExecuteIngestJob(job_repository, runners).execute("ingest-1")["outcome"] == "succeeded"
    assert ExecuteSyncQdrantJob(job_repository, runners).execute("sync-1")["outcome"] == "succeeded"
    assert ExecuteTrainRouterJob(job_repository, runners).execute("train-1")["run_id"] == "router-1"
    assert [job_id for job_id, _ in jobs.succeeded] == ["ingest-1", "sync-1", "train-1"]

    queue = Queue()
    cancelled = CancelJob(job_repository, cast(JobQueue, queue)).execute("job-1")
    assert cancelled.status == "cancelled"
    assert queue.cancelled == ["job-1"]

    uow = RecordingUnitOfWork()

    def factory() -> RecordingUnitOfWork:
        return uow

    manager = SimpleNamespace(promote=lambda run_id, knowledge_base: Path("router.skops"))
    promoted = PromotePolicy(manager, cast(UnitOfWorkFactory, factory)).execute(
        dto.RouterPromoteCommand(run_id="router-1", knowledge_base="demo")
    )
    assert promoted == {"status": "promoted", "path": "router.skops"}
    assert uow.promoted == [("router-1", "demo")]

    trace_store = SimpleNamespace(
        get_trace=lambda run_id: {"run_id": run_id},
        list_events=lambda run_id: [{"run_id": run_id, "sequence": 0}],
    )
    inspector = InspectTrace(trace_store, SimpleNamespace())
    assert inspector.run("run-1") == {"run_id": "run-1"}
    assert inspector.events("run-1")[0]["sequence"] == 0
