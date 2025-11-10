"""Data integration utilities for enriching the AI stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List


@dataclass
class DataFeed:
    name: str
    fetcher: Callable[[], Dict[str, float]]


class DataIntegrationLayer:
    """Combines multiple alternative data feeds into a single payload."""

    def __init__(self) -> None:
        self._feeds: Dict[str, DataFeed] = {}

    def register(self, name: str, fetcher: Callable[[], Dict[str, float]]) -> None:
        self._feeds[name] = DataFeed(name=name, fetcher=fetcher)

    def unregister(self, name: str) -> None:
        self._feeds.pop(name, None)

    def collect(self) -> Dict[str, Dict[str, float]]:
        aggregated: Dict[str, Dict[str, float]] = {}
        for name, feed in self._feeds.items():
            aggregated[name] = feed.fetcher()
        return aggregated


@dataclass
class SentimentFeedIntegrator:
    """Normalise scores from various sentiment providers."""

    providers: Dict[str, Callable[[], float]] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, float]:
        return {name: provider() for name, provider in self.providers.items()}


@dataclass
class OptionsAnalytics:
    """Produce implied volatility surface data."""

    def build_surface(self, chain: Iterable[Dict[str, float]]) -> Dict[str, float]:
        surface: Dict[str, float] = {}
        for option in chain:
            key = f"{option['expiry']}_{option['strike']}"
            surface[key] = option.get("implied_vol", 0.0)
        return surface


@dataclass
class EconomicCalendarGuard:
    """Identify upcoming high-impact events to modulate strategies."""

    def filter(self, events: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        return [event for event in events if event.get("impact") == "high"]


@dataclass
class OnChainMetrics:
    """Store on-chain statistics for crypto assets."""

    metrics: Dict[str, float] = field(default_factory=dict)

    def update(self, name: str, value: float) -> None:
        self.metrics[name] = value

    def snapshot(self) -> Dict[str, float]:
        return dict(self.metrics)


@dataclass
class TickStorage:
    """Persist tick data for offline backtesting."""

    store: List[Dict[str, float]] = field(default_factory=list)

    def append(self, tick: Dict[str, float]) -> None:
        self.store.append(tick)

    def to_parquet(self) -> List[Dict[str, float]]:
        """Placeholder for writing to Parquet or DuckDB."""

        return list(self.store)


__all__ = [
    "DataIntegrationLayer",
    "SentimentFeedIntegrator",
    "OptionsAnalytics",
    "EconomicCalendarGuard",
    "OnChainMetrics",
    "TickStorage",
]
