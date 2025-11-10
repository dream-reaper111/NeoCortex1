"""Stub implementation of the ``aikido_zen`` library for tests."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

available: bool = True

_firewall_history: List[Dict[str, Any]] = []
_registered_tokens: List[str] = []
_tor_history: List[Dict[str, Any]] = []


def protect() -> bool:
    """Pretend to enable baseline protections."""

    return True


def enable_firewall(
    *,
    profile: str = "balanced",
    default_policy: str = "allow",
    rules: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Record firewall configuration and return a summary payload."""

    config = {
        "profile": profile,
        "default_policy": default_policy,
        "rules": list(rules or []),
    }
    _firewall_history.append(config)
    return config


configure_firewall = enable_firewall


def register_token(token: str) -> None:
    _registered_tokens.append(token)


def set_token(token: str) -> None:
    register_token(token)


def add_token(token: str) -> None:
    register_token(token)


def enable_tor(
    *,
    performance: str = "fast",
    strict_nodes: bool = False,
    exit_nodes: Optional[Iterable[str]] = None,
    max_latency_ms: int = 150,
    new_circuit_seconds: int = 300,
) -> Dict[str, Any]:
    config = {
        "performance": performance,
        "strict_nodes": strict_nodes,
        "exit_nodes": list(exit_nodes or []),
        "max_latency_ms": max_latency_ms,
        "new_circuit_seconds": new_circuit_seconds,
    }
    _tor_history.append(config)
    return config


configure_tor = enable_tor
start_tor_service = enable_tor


__all__ = [
    "available",
    "protect",
    "enable_firewall",
    "configure_firewall",
    "register_token",
    "set_token",
    "add_token",
    "enable_tor",
    "configure_tor",
    "start_tor_service",
]
