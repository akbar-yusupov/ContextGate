from __future__ import annotations

from types import TracebackType
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from contextgate.adapters.sqlalchemy import Job
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
    ) -> Job:
        if idempotency_key:
            existing = self.raw_session.scalar(
                select(Job).where(
                    Job.kind == kind,
                    Job.idempotency_key == idempotency_key,
                )
            )
            if existing:
                return existing
        job = Job(
            kind=kind,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        self.raw_session.add(job)
        self.raw_session.flush()
        return job
