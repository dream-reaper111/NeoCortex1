"""Celery application configured for the NeoCortex model orchestration layer."""

from __future__ import annotations

import os

from celery import Celery

from .gpu_scaling import apply_celery_autoscaling

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)

apply_celery_autoscaling()

worker_app = Celery(
    "neocortex",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

worker_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=int(os.getenv("CELERY_PREFETCH_MULTIPLIER", "1")),
)


@worker_app.task(name="healthcheck.ping")
def ping() -> str:
    """Simple ping task used for health checks."""

    return "pong"


__all__ = ["worker_app", "ping"]
