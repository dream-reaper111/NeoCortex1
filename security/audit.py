"""Audit trail helpers for recording security sensitive events."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)


class AuditTrail:
    """Append-only audit log backed by JSON lines."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, actor: str, action: str, details: Dict[str, Any] | None = None) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "details": details or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        logger.info("audit event: %s", payload)

    def stream(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


__all__ = ["AuditTrail"]
