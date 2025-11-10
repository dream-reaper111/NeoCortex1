"""Data integration layer feeding models with alternative data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


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
