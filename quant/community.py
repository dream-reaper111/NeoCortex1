"""Community and collaboration utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Mapping, Sequence


class SignalBroadcaster:
    """Streams signals to registered listeners."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, Callable[[Dict[str, float]], None]] = {}

    def subscribe(self, user_id: str, callback: Callable[[Dict[str, float]], None]) -> None:
        self._subscribers[user_id] = callback

    def unsubscribe(self, user_id: str) -> None:
        self._subscribers.pop(user_id, None)

    def broadcast(self, signal: Dict[str, float]) -> None:
        for callback in self._subscribers.values():
            callback(signal)


@dataclass
class PortfolioView:
    user_id: str
    equity: float
    positions: Dict[str, float]


class MultiUserDashboard:
    """Stores per-user portfolio snapshots for dashboard rendering."""

    def __init__(self) -> None:
        self._views: Dict[str, PortfolioView] = {}

    def update(self, view: PortfolioView) -> None:
        self._views[view.user_id] = view

    def get(self, user_id: str) -> PortfolioView:
        return self._views[user_id]

    def all_views(self) -> List[PortfolioView]:
        return list(self._views.values())


class Leaderboard:
    """Ranks strategies by performance metrics."""

    def rank(self, metrics: Mapping[str, Dict[str, float]], metric: str = "sharpe") -> List[tuple[str, float]]:
        return sorted(((name, data.get(metric, 0.0)) for name, data in metrics.items()), key=lambda item: item[1], reverse=True)


class ChatAndAlerts:
    """Dispatches alerts to external integrations (e.g. Discord/Telegram)."""

    def __init__(self) -> None:
        self._channels: Dict[str, List[str]] = defaultdict(list)

    def register_channel(self, name: str, webhook: str) -> None:
        self._channels[name].append(webhook)

    def broadcast(self, channel: str, message: str) -> List[str]:
        hooks = self._channels.get(channel, [])
        # return list for integration tests to assert; real implementation would POST.
        return [f"POST {hook} -> {message}" for hook in hooks]


class JournalAutomation:
    """Generates trading journal entries in Markdown format."""

    def create_entry(self, date: datetime, summary: str, trades: Sequence[Dict[str, object]]) -> str:
        lines = [f"# Trading Journal - {date:%Y-%m-%d}", "", summary, "", "## Trades"]
        for trade in trades:
            lines.append(f"- {trade['symbol']}: {trade['pnl']:+.2f} ({trade.get('notes', 'no notes')})")
        return "\n".join(lines) + "\n"
