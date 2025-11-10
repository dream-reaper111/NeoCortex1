"""On-chain analytics feed for blockchain metrics."""

from __future__ import annotations

from typing import Any, Dict

import requests


class OnChainAnalyticsFeed:
    def __init__(self, endpoint: str, api_key: str | None = None):
        self.endpoint = endpoint
        self.api_key = api_key

    def fetch(self, metric: str, network: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        response = requests.get(
            self.endpoint,
            params={"metric": metric, "network": network},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()


__all__ = ["OnChainAnalyticsFeed"]
