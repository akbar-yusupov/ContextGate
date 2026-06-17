from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from contextgate.adapters.local.ingestion_service import IngestionService
from contextgate.adapters.mlflow.evaluation_store import BenchmarkService
from contextgate.adapters.mlflow.router_registry import RouterManager
from contextgate.adapters.sqlalchemy import Job, RouterVersion
from contextgate.adapters.sqlalchemy.lookup import get_knowledge_base


class IngestionServiceJobRunner:
    def __init__(self, service: IngestionService, session_factory: sessionmaker) -> None:
        self.service = service
        self.session_factory = session_factory

    def ingest(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            return self.service.ingest_path(
                session,
                payload["knowledge_base"],
                Path(payload["path"]),
                metadata=payload.get("metadata"),
                job=job,
            )

    def sync_qdrant(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            return self.service.sync_collection(
                session,
                payload["knowledge_base"],
                payload["source_collection"],
                job=job,
            )


class BenchmarkServiceJobRunner:
    def __init__(self, service: BenchmarkService, session_factory: sessionmaker) -> None:
        self.service = service
        self.session_factory = session_factory

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.session_factory() as session:
            return self.service.run(
                session,
                payload["knowledge_base"],
                Path(payload["dataset_path"]),
                payload.get("policies"),
                payload.get("evaluate_answers", False),
            )


class RouterTrainingServiceJobRunner:
    def __init__(self, manager: RouterManager, session_factory: sessionmaker) -> None:
        self.manager = manager
        self.session_factory = session_factory

    def train(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.manager.train(payload["benchmark_run_id"], payload["knowledge_base"])
        with self.session_factory() as session:
            knowledge_base = get_knowledge_base(session, payload["knowledge_base"])
            version = session.scalar(
                select(RouterVersion).where(
                    RouterVersion.knowledge_base_id == knowledge_base.id,
                    RouterVersion.run_id == result["run_id"],
                )
            )
            if version is None:
                version = RouterVersion(
                    knowledge_base_id=knowledge_base.id,
                    run_id=result["run_id"],
                    artifact_path=result["artifact_path"],
                    status="candidate",
                    metrics_json=result["metrics"],
                )
            else:
                version.artifact_path = result["artifact_path"]
                version.status = "candidate"
                version.metrics_json = result["metrics"]
            session.add(version)
            session.commit()
        return result
