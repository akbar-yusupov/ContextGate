from __future__ import annotations

from celery import Celery

from contextgate.config import get_settings

settings = get_settings()
celery_app = Celery(
    "contextgate",
    broker=settings.resolved_redis_url,
    backend=settings.resolved_redis_url,
)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_soft_time_limit=settings.worker_task_soft_time_limit_seconds,
    task_time_limit=settings.worker_task_time_limit_seconds,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
