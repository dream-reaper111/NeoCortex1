"""Sentiment analysis feed aggregator."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import requests


class SentimentFeed:
    def __init__(self, providers: Iterable[str]):
        self.providers = list(providers)

    def fetch(self, symbol: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for provider in self.providers:
            response = requests.get(provider, params={"symbol": symbol}, timeout=10)
            response.raise_for_status()
            payload = response.json()
            payload.setdefault("provider", provider)
            results.append(payload)
        return results


__all__ = ["SentimentFeed"]
