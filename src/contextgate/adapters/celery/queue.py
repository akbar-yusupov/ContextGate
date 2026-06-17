from __future__ import annotations

from celery import Celery


class CeleryJobQueue:
    TASK_NAMES = {
        "ingest": "contextgate.ingest",
        "benchmark": "contextgate.adapters.mlflow.evaluation_store",
        "sync_qdrant": "contextgate.sync_qdrant",
        "router_train": "contextgate.adapters.mlflow.router_registry_train",
    }

    def __init__(self, celery_app: Celery) -> None:
        self.celery_app = celery_app

    def enqueue(self, kind: str, job_id: str) -> None:
        task_name = self.TASK_NAMES[kind]
        self.celery_app.send_task(task_name, args=[job_id])
