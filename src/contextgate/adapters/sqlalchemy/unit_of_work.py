from __future__ import annotations

from datetime import UTC, datetime
from types import TracebackType
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from contextgate.adapters.sqlalchemy import Job, JobOutbox, KnowledgeBase, RouterVersion
from contextgate.domain.errors import ContextGateError
from contextgate.ports.repositories import UnitOfWork


class SqlAlchemyUnitOfWorkFactory:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def __call__(self) -> UnitOfWork:
        return SqlAlchemyUnitOfWork(self.session_factory)


class SqlAlchemyUnitOfWork:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> UnitOfWork:
        self._session = self.session_factory()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is None:
            return
        try:
            if exc_type is not None:
                self._session.rollback()
        finally:
            self._session.close()

    @property
    def raw_session(self) -> Session:
        if self._session is None:
            raise RuntimeError("UnitOfWork is not active")
        return self._session

    def commit(self) -> None:
        self.raw_session.commit()

    def rollback(self) -> None:
        self.raw_session.rollback()

    def create_job(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> tuple[Job, bool]:
        if idempotency_key:
            existing = self.raw_session.scalar(
                select(Job).where(
                    Job.kind == kind,
                    Job.idempotency_key == idempotency_key,
                )
            )
            if existing:
                if existing.payload != payload:
                    raise ContextGateError(
                        "policy_rejected",
                        "Idempotency key was already used with a different payload.",
                        {"kind": kind, "idempotency_key": idempotency_key},
                    )
                return existing, False
        job = Job(
            kind=kind,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        self.raw_session.add(job)
        self.raw_session.flush()
        self.raw_session.add(JobOutbox(job_id=job.id, kind=kind))
        return job, True

    def mark_job_enqueued(self, job_id: str) -> None:
        outbox = self.raw_session.scalar(select(JobOutbox).where(JobOutbox.job_id == job_id))
        if outbox is None:
            raise ValueError(f"Job outbox record not found: {job_id}")
        outbox.status = "dispatched"
        outbox.dispatched_at = datetime.now(UTC)
        self.raw_session.flush()

    def pending_job_dispatches(self) -> list[tuple[str, str]]:
        rows = self.raw_session.scalars(
            select(JobOutbox).where(JobOutbox.status == "pending").order_by(JobOutbox.created_at)
        ).all()
        return [(row.kind, row.job_id) for row in rows]

    def promote_router_version(self, run_id: str, knowledge_base: str) -> None:
        kb = self.raw_session.scalar(
            select(KnowledgeBase).where(
                (KnowledgeBase.slug == knowledge_base) | (KnowledgeBase.id == knowledge_base)
            )
        )
        if kb is None:
            raise ValueError(f"Knowledge base not found: {knowledge_base}")
        candidate = self.raw_session.scalar(
            select(RouterVersion).where(
                RouterVersion.knowledge_base_id == kb.id,
                RouterVersion.run_id == run_id,
            )
        )
        if candidate is None:
            raise ValueError(f"Router version not found: {run_id}")
        for active in self.raw_session.scalars(
            select(RouterVersion).where(
                RouterVersion.knowledge_base_id == kb.id,
                RouterVersion.status == "active",
            )
        ):
            active.status = "archived"
        candidate.status = "active"
        self.raw_session.flush()
