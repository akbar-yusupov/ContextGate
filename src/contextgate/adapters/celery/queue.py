from __future__ import annotations

from celery import Celery


class CeleryJobQueue:
    TASK_NAMES = {
        "ingest": "contextgate.ingest",
        "benchmark": "contextgate.benchmark",
        "sync_qdrant": "contextgate.sync_qdrant",
        "router_train": "contextgate.router_train",
    }

    def __init__(self, celery_app: Celery) -> None:
        self.celery_app = celery_app

    def enqueue(self, kind: str, job_id: str) -> None:
        task_name = self.TASK_NAMES[kind]
        self.celery_app.send_task(task_name, args=[job_id], task_id=job_id)

    def cancel(self, job_id: str) -> None:
        self.celery_app.control.revoke(job_id, terminate=False)
