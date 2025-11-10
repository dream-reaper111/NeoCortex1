"""Community and social trading features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class SignalStreamer:
    """Fan out trading signals to connected subscribers."""

    subscribers: List[Callable[[Dict[str, float]], None]] = field(default_factory=list)

    def broadcast(self, signal: Dict[str, float]) -> None:
        for subscriber in self.subscribers:
            subscriber(signal)


@dataclass
class MultiUserDashboard:
    """Store per-user portfolio states for dashboard rendering."""

    portfolios: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def snapshot(self, user: str) -> Dict[str, float]:
        return dict(self.portfolios.get(user, {}))


@dataclass
class Leaderboard:
    """Rank strategies by an arbitrary performance metric."""

    def rank(self, metrics: Dict[str, float]) -> List[tuple[str, float]]:
        return sorted(metrics.items(), key=lambda item: item[1], reverse=True)


@dataclass
class ChatAlerts:
    """Dispatch alerts to chat integrations such as Discord or Telegram."""

    webhook_senders: Dict[str, Callable[[str], None]] = field(default_factory=dict)

    def notify(self, message: str) -> None:
        for sender in self.webhook_senders.values():
            sender(message)


@dataclass
class JournalAutomation:
    """Create structured journal entries after each trading session."""

    formatter: Callable[[Dict[str, float]], str]
    sink: Callable[[str], None]

    def create_entry(self, stats: Dict[str, float]) -> str:
        entry = self.formatter(stats)
        self.sink(entry)
        return entry
