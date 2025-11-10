"""Options chain data client."""

from __future__ import annotations

from typing import Any, Dict, List

import requests


class OptionsChainFeed:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def fetch(self, symbol: str, expiry: str | None = None) -> Dict[str, Any]:
        params = {"symbol": symbol}
        if expiry:
            params["expiry"] = expiry
        response = requests.get(self.endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()


__all__ = ["OptionsChainFeed"]
