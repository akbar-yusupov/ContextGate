from __future__ import annotations

import builtins
import hashlib
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from contextgate.adapters.sqlalchemy import (
    ApiKey,
    Document,
    GatewayPolicy,
    Job,
    KnowledgeBase,
    RouterVersion,
)
from contextgate.domain.errors import ContextGateError


class SqlAlchemyKnowledgeBaseRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory
        self._cache: dict[str, tuple[float, KnowledgeBase]] = {}
        self._cache_lock = Lock()

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
            with self._cache_lock:
                self._cache.clear()
            return kb

    def list(self) -> list[KnowledgeBase]:
        with self.session_factory() as session:
            return list(session.scalars(select(KnowledgeBase)).all())

    def get(self, identifier: str) -> KnowledgeBase:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(identifier)
            if cached is not None and cached[0] > now:
                return cached[1]
        with self.session_factory() as session:
            kb = session.scalar(
                select(KnowledgeBase).where(
                    (KnowledgeBase.slug == identifier) | (KnowledgeBase.id == identifier)
                )
            )
            if kb is None:
                raise ValueError(f"Knowledge base not found: {identifier}")
            session.expunge(kb)
            with self._cache_lock:
                expires_at = time.monotonic() + 1
                self._cache[identifier] = (expires_at, kb)
                self._cache[kb.id] = (expires_at, kb)
                self._cache[kb.slug] = (expires_at, kb)
            return kb

    def get_job(self, job_id: str) -> Job:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError("Job not found")
            session.expunge(job)
            return job

    def list_documents(self, identifier: str) -> builtins.list[Document]:
        with self.session_factory() as session:
            kb = session.scalar(
                select(KnowledgeBase).where(
                    (KnowledgeBase.slug == identifier) | (KnowledgeBase.id == identifier)
                )
            )
            if kb is None:
                raise ValueError(f"Knowledge base not found: {identifier}")
            return list(
                session.scalars(
                    select(Document)
                    .where(Document.knowledge_base_id == kb.id)
                    .order_by(Document.created_at.desc())
                ).all()
            )


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
            if job.status == "cancelled":
                raise ContextGateError("policy_rejected", "Cancelled job cannot be started")
            if job.status in {"succeeded", "succeeded_with_errors"}:
                raise ContextGateError(
                    "policy_rejected",
                    f"Terminal job cannot be started again: {job.status}",
                )
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
            if job.status == "cancelled":
                return
            job.status = (
                "succeeded_with_errors"
                if result.get("outcome") == "succeeded_with_errors"
                else "succeeded"
            )
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
            if job.status == "cancelled":
                return
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

    def cancel(self, job_id: str) -> Job:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError(f"Job not found: {job_id}")
            if job.status in {"succeeded", "succeeded_with_errors", "failed"}:
                raise ContextGateError(
                    "policy_rejected",
                    f"Terminal job cannot be cancelled: {job.status}",
                )
            job.status = "cancelled"
            job.finished_at = datetime.now(UTC)
            session.add(job)
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job


class SqlAlchemyRouterVersionRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def active_artifact(self, knowledge_base: str):
        with self.session_factory() as session:
            row = session.scalar(
                select(RouterVersion)
                .join(KnowledgeBase, KnowledgeBase.id == RouterVersion.knowledge_base_id)
                .where(
                    (KnowledgeBase.slug == knowledge_base) | (KnowledgeBase.id == knowledge_base),
                    RouterVersion.status == "active",
                )
                .order_by(RouterVersion.created_at.desc())
            )
            return (Path(row.artifact_path), row.artifact_checksum) if row else None

    def list(self, knowledge_base: str) -> list[RouterVersion]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(RouterVersion)
                .join(KnowledgeBase, KnowledgeBase.id == RouterVersion.knowledge_base_id)
                .where(
                    (KnowledgeBase.slug == knowledge_base) | (KnowledgeBase.id == knowledge_base)
                )
                .order_by(RouterVersion.created_at.desc())
            )
            return list(rows.all())


class SqlAlchemyApiKeyRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    @staticmethod
    def _secret() -> str:
        return f"ctxg_{secrets.token_urlsafe(32)}"

    def create(self, name: str, scopes: list[str]) -> tuple[ApiKey, str]:
        secret = self._secret()
        with self.session_factory() as session:
            if session.scalar(select(ApiKey).where(ApiKey.name == name)):
                raise ContextGateError("policy_rejected", "API key name already exists")
            record = ApiKey(
                name=name,
                key_hash=hashlib.sha256(secret.encode()).hexdigest(),
                scopes_json=scopes,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record, secret

    def list(self) -> list[ApiKey]:
        with self.session_factory() as session:
            return list(session.scalars(select(ApiKey).order_by(ApiKey.created_at.desc())).all())

    def rotate(self, key_id: str) -> tuple[ApiKey, str]:
        secret = self._secret()
        with self.session_factory() as session:
            record = session.get(ApiKey, key_id)
            if record is None:
                raise ValueError("API key not found")
            record.key_hash = hashlib.sha256(secret.encode()).hexdigest()
            record.enabled = True
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record, secret

    def disable(self, key_id: str) -> ApiKey:
        with self.session_factory() as session:
            record = session.get(ApiKey, key_id)
            if record is None:
                raise ValueError("API key not found")
            record.enabled = False
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record


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

    def list(self) -> list[GatewayPolicy]:
        with self.session_factory() as session:
            return list(
                session.scalars(
                    select(GatewayPolicy).order_by(GatewayPolicy.created_at.desc())
                ).all()
            )

    def resolve_active(self, policy_id: str) -> GatewayPolicy:
        policy = self.get(policy_id)
        if policy.status != "active":
            raise ContextGateError(
                "policy_rejected",
                "Gateway policy must be promoted before it can serve traffic.",
                {"policy_id": policy_id, "status": policy.status},
            )
        return policy

    def promote(self, policy_id: str) -> GatewayPolicy:
        from datetime import UTC, datetime

        with self.session_factory() as session:
            policy = session.get(GatewayPolicy, policy_id)
            if policy is None:
                raise ValueError("Policy not found")
            for active in session.scalars(
                select(GatewayPolicy).where(
                    GatewayPolicy.status == "active",
                    GatewayPolicy.id != policy_id,
                )
            ):
                active.status = "archived"
            policy.status = "active"
            policy.promoted_at = datetime.now(UTC)
            session.add(policy)
            session.commit()
            session.refresh(policy)
            session.expunge(policy)
            return policy
