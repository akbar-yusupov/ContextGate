from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from contextgate.domain.models import CostRecord, ProviderCall


class KnowledgeBaseRepository(Protocol):
    def create(self, payload: Any) -> Any: ...

    def list(self) -> list[Any]: ...

    def get(self, identifier: str) -> Any: ...

    def get_job(self, job_id: str) -> Any: ...


class UnitOfWork(Protocol):
    @property
    def raw_session(self) -> Any: ...

    def __enter__(self) -> UnitOfWork: ...

    def __exit__(self, exc_type, exc, traceback) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def create_job(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> Any: ...


class UnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


class JobRepository(Protocol):
    def create(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> Any: ...

    def start(self, job_id: str) -> Any: ...

    def succeed(self, job_id: str, result: dict[str, Any]) -> None: ...

    def fail(self, job_id: str, error: str, details: dict[str, Any] | None = None) -> None: ...

    def set_progress(self, job_id: str, progress: float) -> None: ...


class JobQueue(Protocol):
    def enqueue(self, kind: str, job_id: str) -> None: ...


class IngestionJobRunner(Protocol):
    def ingest(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def sync_qdrant(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class BenchmarkJobRunner(Protocol):
    def run(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class RouterTrainingJobRunner(Protocol):
    def train(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class CostLedger(Protocol):
    def record(self, record: CostRecord) -> None: ...

    def list_for_run(self, run_id: str) -> list[CostRecord]: ...


class TraceStore(Protocol):
    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None: ...

    def list_events(self, run_id: str) -> list[dict[str, Any]]: ...

    def get_trace(self, run_id: str) -> dict[str, Any]: ...


class ProviderRegistry(Protocol):
    def list(self) -> list[dict[str, Any]]: ...

    def test(self, provider: str | None = None) -> dict[str, Any]: ...

    def choose(
        self,
        *,
        cost_budget_usd: float | None,
        latency_budget_ms: float,
        allowed_providers: Sequence[str] | None = None,
        requested_provider: str | None = None,
    ) -> str: ...


class ResponseCache(Protocol):
    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any) -> None: ...


class RouterRepository(Protocol):
    def train(self, benchmark_run_id: str, knowledge_base: str) -> dict[str, Any]: ...

    def promote(self, benchmark_run_id: str, knowledge_base: str) -> Path: ...


class LLMProvider(Protocol):
    def generate(
        self,
        *,
        query: str,
        contexts: list[str],
        system_prompt: str | None = None,
    ) -> tuple[str, ProviderCall]: ...
