from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from contextgate.adapters.sqlalchemy import GatewayPolicy, Job, KnowledgeBase


class SqlAlchemyKnowledgeBaseRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def create(self, payload: Any) -> KnowledgeBase:
        with self.session_factory() as session:
            existing = session.scalar(
                select(KnowledgeBase).where(KnowledgeBase.slug == payload.slug)
            )
            if existing:
                raise ValueError("Knowledge base slug already exists")
            kb = KnowledgeBase(
                name=payload.name,
                slug=payload.slug,
                description=payload.description,
                collection_name=f"contextgate-{payload.slug}",
            )
            session.add(kb)
            session.commit()
            session.refresh(kb)
            return kb

    def list(self) -> list[KnowledgeBase]:
        with self.session_factory() as session:
            return list(session.scalars(select(KnowledgeBase)).all())

    def get(self, identifier: str) -> KnowledgeBase:
        with self.session_factory() as session:
            kb = session.scalar(
                select(KnowledgeBase).where(
                    (KnowledgeBase.slug == identifier) | (KnowledgeBase.id == identifier)
                )
            )
            if kb is None:
                raise ValueError(f"Knowledge base not found: {identifier}")
            session.expunge(kb)
            return kb

    def get_job(self, job_id: str) -> Job:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError("Job not found")
            session.expunge(job)
            return job


class SqlAlchemyJobRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def create(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> Job:
        with self.session_factory() as session:
            if idempotency_key:
                existing = session.scalar(
                    select(Job).where(
                        Job.kind == kind,
                        Job.idempotency_key == idempotency_key,
                    )
                )
                if existing:
                    session.expunge(existing)
                    return existing
            job = Job(kind=kind, payload=payload, idempotency_key=idempotency_key)
            session.add(job)
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def start(self, job_id: str) -> Job:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError(f"Job not found: {job_id}")
            job.status = "running"
            job.started_at = datetime.now(UTC)
            job.retry_count += 1
            session.add(job)
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def succeed(self, job_id: str, result: dict[str, Any]) -> None:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError(f"Job not found: {job_id}")
            job.status = "succeeded"
            job.progress = 1
            job.result = result
            job.finished_at = datetime.now(UTC)
            session.add(job)
            session.commit()

    def fail(self, job_id: str, error: str, details: dict[str, Any] | None = None) -> None:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError(f"Job not found: {job_id}")
            job.status = "failed"
            job.error = error
            job.error_json = details or {"message": error}
            job.finished_at = datetime.now(UTC)
            session.add(job)
            session.commit()

    def set_progress(self, job_id: str, progress: float) -> None:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError(f"Job not found: {job_id}")
            job.progress = progress
            session.add(job)
            session.commit()


class SqlAlchemyPolicyRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def create(self, payload: Any) -> GatewayPolicy:
        with self.session_factory() as session:
            existing = session.scalar(
                select(GatewayPolicy).where(GatewayPolicy.name == payload.name)
            )
            if existing:
                raise ValueError("Gateway policy name already exists")
            policy = GatewayPolicy(
                name=payload.name,
                description=payload.description,
                retrieval_policy=payload.retrieval_policy,
                provider_policy=payload.provider_policy,
                latency_budget_ms=payload.latency_budget_ms,
                cost_budget_usd=payload.cost_budget_usd,
                status="draft",
            )
            session.add(policy)
            session.commit()
            session.refresh(policy)
            session.expunge(policy)
            return policy

    def get(self, policy_id: str) -> GatewayPolicy:
        with self.session_factory() as session:
            policy = session.get(GatewayPolicy, policy_id)
            if policy is None:
                raise ValueError("Policy not found")
            session.expunge(policy)
            return policy

    def promote(self, policy_id: str) -> GatewayPolicy:
        from datetime import UTC, datetime

        with self.session_factory() as session:
            policy = session.get(GatewayPolicy, policy_id)
            if policy is None:
                raise ValueError("Policy not found")
            policy.status = "active"
            policy.promoted_at = datetime.now(UTC)
            session.add(policy)
            session.commit()
            session.refresh(policy)
            session.expunge(policy)
            return policy
