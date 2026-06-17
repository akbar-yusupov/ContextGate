from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from contextgate.application.use_cases import ExecuteBenchmarkJob


@dataclass
class FakeJob:
    id: str
    payload: dict[str, Any]
    retry_count: int = 0
    status: str = "queued"
    result: dict[str, Any] | None = None
    error_json: dict[str, Any] | None = None


class FakeJobRepository:
    def __init__(self) -> None:
        self.job = FakeJob("job-1", {"knowledge_base": "demo"})
        self.failures: list[dict[str, Any]] = []

    def start(self, job_id: str) -> FakeJob:
        assert job_id == self.job.id
        self.job.retry_count += 1
        self.job.status = "running"
        return self.job

    def succeed(self, job_id: str, result: dict[str, Any]) -> None:
        assert job_id == self.job.id
        self.job.status = "succeeded"
        self.job.result = result

    def fail(self, job_id: str, error: str, details: dict[str, Any] | None = None) -> None:
        assert job_id == self.job.id
        self.job.status = "failed"
        self.job.error_json = details
        self.failures.append({"error": error, "details": details})

    def create(self, **kwargs):  # pragma: no cover - unused protocol member
        raise NotImplementedError

    def set_progress(self, job_id: str, progress: float) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class FakeBenchmarkRunner:
    result: dict[str, Any] = field(default_factory=lambda: {"run_id": "bench-1"})
    fail: bool = False

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("benchmark failed")
        return self.result


def test_job_execution_records_success_and_retry_count() -> None:
    jobs = FakeJobRepository()
    use_case = ExecuteBenchmarkJob(jobs, FakeBenchmarkRunner())

    result = use_case.execute("job-1")

    assert result == {"run_id": "bench-1"}
    assert jobs.job.status == "succeeded"
    assert jobs.job.retry_count == 1


def test_job_execution_preserves_structured_failure() -> None:
    jobs = FakeJobRepository()
    use_case = ExecuteBenchmarkJob(jobs, FakeBenchmarkRunner(fail=True))

    with pytest.raises(RuntimeError, match="benchmark failed"):
        use_case.execute("job-1")

    assert jobs.job.status == "failed"
    assert jobs.job.retry_count == 1
    assert jobs.job.error_json == {"type": "RuntimeError", "message": "benchmark failed"}
