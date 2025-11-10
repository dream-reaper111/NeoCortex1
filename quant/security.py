"""System and security enhancements for NeoCortex."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List

from cryptography.fernet import Fernet


class APIKeyVault:
    """Encrypts and stores broker API keys using Fernet symmetric encryption."""

    def __init__(self, master_key: bytes | None = None) -> None:
        self._master_key = master_key or Fernet.generate_key()
        self._fernet = Fernet(self._master_key)
        self._store: Dict[str, bytes] = {}

    @property
    def master_key(self) -> bytes:
        return self._master_key

    def put(self, broker: str, api_key: str) -> None:
        token = self._fernet.encrypt(api_key.encode("utf-8"))
        self._store[broker] = token

    def get(self, broker: str) -> str:
        token = self._store.get(broker)
        if token is None:
            raise KeyError(broker)
        return self._fernet.decrypt(token).decode("utf-8")

    def remove(self, broker: str) -> None:
        self._store.pop(broker, None)


class AccessControlList:
    """Light-weight role based access control implementation."""

    def __init__(self) -> None:
        self._roles: Dict[str, set[str]] = {
            "admin": {"read", "write", "execute"},
            "analyst": {"read", "execute"},
            "client": {"read"},
        }
        self._assignments: Dict[str, str] = {}

    def assign(self, user_id: str, role: str) -> None:
        if role not in self._roles:
            raise ValueError(f"Unknown role '{role}'")
        self._assignments[user_id] = role

    def permissions(self, user_id: str) -> set[str]:
        role = self._assignments.get(user_id, "client")
        return set(self._roles[role])

    def check(self, user_id: str, permission: str) -> bool:
        return permission in self.permissions(user_id)


@dataclass
class AuditEvent:
    timestamp: datetime
    actor: str
    action: str
    payload: Dict[str, object]


class AuditLog:
    """Captures webhook activity and surfaces anomalies."""

    def __init__(self, anomaly_threshold: float = 3.0) -> None:
        self._events: List[AuditEvent] = []
        self.anomaly_threshold = anomaly_threshold

    def record(self, actor: str, action: str, payload: Dict[str, object]) -> AuditEvent:
        event = AuditEvent(timestamp=datetime.utcnow(), actor=actor, action=action, payload=payload)
        self._events.append(event)
        return event

    def to_json(self) -> str:
        serialisable = [
            {
                "timestamp": event.timestamp.isoformat() + "Z",
                "actor": event.actor,
                "action": event.action,
                "payload": event.payload,
            }
            for event in self._events
        ]
        return json.dumps(serialisable, indent=2)

    def detect_anomalies(self) -> List[AuditEvent]:
        if not self._events:
            return []
        sorted_events = sorted(self._events, key=lambda ev: ev.timestamp)
        durations: List[float] = []
        for prev, current in zip(sorted_events, sorted_events[1:]):
            durations.append((current.timestamp - prev.timestamp).total_seconds())
        sizes = [float(event.payload.get("size", 0.0)) for event in sorted_events]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        avg_volume = sum(sizes) / len(sizes)
        anomalies: List[AuditEvent] = []
        for event, size in zip(sorted_events, sizes):
            if avg_volume and size > avg_volume * self.anomaly_threshold:
                anomalies.append(event)
        for idx, duration in enumerate(durations):
            if avg_duration and duration > avg_duration * self.anomaly_threshold:
                anomalies.append(sorted_events[idx + 1])
        deduped: Dict[str, AuditEvent] = {}
        for event in anomalies:
            deduped.setdefault(event.timestamp.isoformat(), event)
        return sorted(deduped.values(), key=lambda ev: ev.timestamp)


@dataclass
class ProcessStatus:
    name: str
    healthy: bool
    latency_ms: float
    details: Dict[str, object] = field(default_factory=dict)


class SystemHealthMonitor:
    """Tracks infrastructure health and executes self-healing callbacks."""

    def __init__(self) -> None:
        self._statuses: Dict[str, ProcessStatus] = {}
        self._remediation: Dict[str, Callable[[ProcessStatus], None]] = {}
        self._observers: List[Callable[[ProcessStatus], None]] = []

    def update_status(self, status: ProcessStatus) -> None:
        self._statuses[status.name] = status
        if not status.healthy and status.name in self._remediation:
            self._remediation[status.name](status)
        for observer in self._observers:
            observer(status)

    def register_remediation(self, process: str, callback: Callable[[ProcessStatus], None]) -> None:
        self._remediation[process] = callback

    def register_observer(self, callback: Callable[[ProcessStatus], None]) -> None:
        self._observers.append(callback)

    def summary(self) -> Dict[str, Dict[str, object]]:
        return {name: {"healthy": status.healthy, "latency_ms": status.latency_ms, "details": status.details} for name, status in self._statuses.items()}

    def unhealthy_processes(self) -> List[ProcessStatus]:
        return [status for status in self._statuses.values() if not status.healthy]
