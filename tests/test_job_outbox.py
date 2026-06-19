from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextgate.adapters.sqlalchemy.models import Base
from contextgate.adapters.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWorkFactory
from contextgate.application.use_cases import DispatchPendingJobs, IngestDocuments
from contextgate.domain.errors import ContextGateError


class RecordingQueue:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple[str, str]] = []

    def enqueue(self, kind: str, job_id: str) -> None:
        if self.fail:
            raise ConnectionError("broker unavailable")
        self.sent.append((kind, job_id))

    def cancel(self, job_id: str) -> None:
        return None


def _factory(tmp_path: Path) -> SqlAlchemyUnitOfWorkFactory:
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    Base.metadata.create_all(engine)
    return SqlAlchemyUnitOfWorkFactory(sessionmaker(bind=engine, expire_on_commit=False))


def test_idempotent_job_is_published_only_once(tmp_path: Path) -> None:
    factory = _factory(tmp_path)
    queue = RecordingQueue()
    use_case = IngestDocuments(factory, queue)

    first = use_case.enqueue(
        knowledge_base="demo",
        path=Path("first.md"),
        idempotency_key="same-key",
    )
    second = use_case.enqueue(
        knowledge_base="demo",
        path=Path("first.md"),
        idempotency_key="same-key",
    )

    assert first.id == second.id
    assert queue.sent == [("ingest", first.id)]


def test_idempotency_key_rejects_a_different_payload(tmp_path: Path) -> None:
    factory = _factory(tmp_path)
    use_case = IngestDocuments(factory, RecordingQueue())
    use_case.enqueue(
        knowledge_base="demo",
        path=Path("first.md"),
        idempotency_key="same-key",
    )

    with pytest.raises(ContextGateError, match="different payload"):
        use_case.enqueue(
            knowledge_base="demo",
            path=Path("second.md"),
            idempotency_key="same-key",
        )


def test_pending_outbox_is_replayed_after_broker_recovery(tmp_path: Path) -> None:
    factory = _factory(tmp_path)
    with pytest.raises(ConnectionError, match="broker unavailable"):
        IngestDocuments(factory, RecordingQueue(fail=True)).enqueue(
            knowledge_base="demo",
            path=Path("first.md"),
            idempotency_key="outage-key",
        )

    recovered_queue = RecordingQueue()
    dispatched = DispatchPendingJobs(factory, recovered_queue).execute()

    assert dispatched == 1
    assert recovered_queue.sent[0][0] == "ingest"
    assert DispatchPendingJobs(factory, recovered_queue).execute() == 0
