from __future__ import annotations

from contextgate.adapters.celery.app import celery_app
from contextgate.apps.container import get_container


@celery_app.task(
    name="contextgate.ingest",
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def ingest_task(self, job_id: str) -> dict:
    return get_container().execute_ingest_job.execute(job_id)


@celery_app.task(
    name="contextgate.benchmark",
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def benchmark_task(self, job_id: str) -> dict:
    return get_container().execute_benchmark_job.execute(job_id)


@celery_app.task(
    name="contextgate.sync_qdrant",
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def sync_qdrant_task(self, job_id: str) -> dict:
    return get_container().execute_sync_qdrant_job.execute(job_id)


@celery_app.task(
    name="contextgate.router_train",
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def train_router_task(self, job_id: str) -> dict:
    return get_container().execute_train_router_job.execute(job_id)
