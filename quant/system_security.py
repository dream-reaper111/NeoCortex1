"""System and security enhancements for the automation platform."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from cryptography.fernet import Fernet


@dataclass
class APIKeyVault:
    """Encrypt and store broker API keys using Fernet symmetric encryption."""

    master_key: bytes = field(default_factory=Fernet.generate_key)
    _fernet: Fernet = field(init=False)
    _store: Dict[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._fernet = Fernet(self.master_key)

    def put(self, identifier: str, plaintext_key: str) -> None:
        self._store[identifier] = self._fernet.encrypt(plaintext_key.encode("utf-8"))

    def get(self, identifier: str) -> Optional[str]:
        token = self._store.get(identifier)
        if token is None:
            return None
        return self._fernet.decrypt(token).decode("utf-8")


@dataclass
class AccessControl:
    """Simple role-based access control registry."""

    role_matrix: Dict[str, List[str]] = field(default_factory=lambda: {
        "admin": ["trade", "configure", "view"],
        "analyst": ["view", "backtest"],
        "client": ["view"],
    })

    def allowed(self, role: str, permission: str) -> bool:
        return permission in self.role_matrix.get(role, [])


@dataclass
class AuditLog:
    """Track webhook events and flag anomalies."""

    max_entries: int = 1000
    history: List[Dict[str, float]] = field(default_factory=list)
    size_threshold: float = 10.0
    night_hours: range = range(0, 6)

    def record(self, timestamp: float, size: float) -> Optional[str]:
        entry = {"timestamp": timestamp, "size": size}
        self.history.append(entry)
        if len(self.history) > self.max_entries:
            self.history.pop(0)

        if abs(size) > self.size_threshold:
            return "size_anomaly"
        hour = time.gmtime(timestamp).tm_hour
        if hour in self.night_hours and abs(size) > self.size_threshold / 2:
            return "time_anomaly"
        return None


@dataclass
class HealthMonitor:
    """Aggregate health metrics for critical services."""

    metrics: Dict[str, float] = field(default_factory=dict)

    def update(self, name: str, value: float) -> None:
        self.metrics[name] = value

    def snapshot(self) -> Dict[str, float]:
        return dict(self.metrics)


@dataclass
class SelfHealingSupervisor:
    """Restart critical processes if metrics fall outside thresholds."""

    restart_callbacks: Dict[str, Callable[[], None]] = field(default_factory=dict)
    latency_threshold_ms: float = 1000.0
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def evaluate(self, metric_name: str, value: float) -> None:
        if metric_name == "latency" and value > self.latency_threshold_ms:
            self._restart("latency")

    def process_failure(self, process_name: str) -> None:
        self._restart(process_name)

    def _restart(self, key: str) -> None:
        callback = self.restart_callbacks.get(key)
        if callback:
            self.logger.warning("Restarting component due to %s condition", key)
            callback()
        else:
            self.logger.error("No restart handler registered for %s", key)
