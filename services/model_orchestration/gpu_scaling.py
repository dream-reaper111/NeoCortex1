"""GPU auto-discovery utilities and Celery scaling hooks."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUDevice:
    """Representation of a single GPU device."""

    id: str
    name: str
    memory_total_mb: int

    @property
    def is_available(self) -> bool:
        return self.memory_total_mb > 0


def _parse_nvidia_smi(output: str) -> List[GPUDevice]:
    devices: List[GPUDevice] = []
    for line in output.splitlines():
        parts = [segment.strip() for segment in line.split(",")]
        if len(parts) < 3:
            continue
        identifier, name, memory = parts[:3]
        try:
            memory_mb = int(memory.split()[0])
        except (ValueError, IndexError):
            memory_mb = 0
        devices.append(GPUDevice(id=identifier, name=name, memory_total_mb=memory_mb))
    return devices


def discover_gpus() -> List[GPUDevice]:
    """Return a list of discovered GPUs using ``nvidia-smi`` when available."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        logger.debug("nvidia-smi not installed; assuming no GPUs present")
        return []
    except subprocess.CalledProcessError as exc:  # pragma: no cover - logging path
        logger.warning("nvidia-smi call failed: %s", exc.stderr)
        return []
    return _parse_nvidia_smi(result.stdout)


def desired_worker_concurrency(gpus: Iterable[GPUDevice], default: int = 1) -> int:
    """Determine desired concurrency for workers based on GPU count."""

    gpus_list = list(gpus)
    if not gpus_list:
        return default
    return max(len(gpus_list), default)


def apply_celery_autoscaling(concurrency: Optional[int] = None) -> None:
    """Apply autoscaling hints for Celery workers using environment variables."""

    if concurrency is None:
        concurrency = desired_worker_concurrency(discover_gpus())
    os.environ.setdefault("CELERYD_CONCURRENCY", str(concurrency))
    os.environ.setdefault("CELERY_ACKS_LATE", "True")
    os.environ.setdefault("CELERY_PREFETCH_MULTIPLIER", "1")
    logger.info("Configured Celery concurrency to %s based on GPU discovery", concurrency)


__all__ = [
    "GPUDevice",
    "discover_gpus",
    "desired_worker_concurrency",
    "apply_celery_autoscaling",
]
