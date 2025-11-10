"""OANDA REST connector."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import requests

from .base import BrokerConnector

_OANDA_API = "https://api-fxtrade.oanda.com/v3"


class OANDAConnector(BrokerConnector):
    name = "oanda"

    @classmethod
    def required_keys(cls) -> Iterable[str]:
        return ["account_id", "token"]

    def _request(self, method: str, path: str, json_body: Dict[str, Any] | None = None, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = f"{_OANDA_API}{path}"
        headers = {"Authorization": f"Bearer {self.credentials['token']}"}
        response = requests.request(method, url, json=json_body, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def place_order(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        payload = {
            "order": {
                "instrument": symbol,
                "units": str(quantity if side.upper() == "BUY" else -quantity),
                "type": "MARKET",
                "timeInForce": "FOK",
            }
        }
        return self._request("POST", f"/accounts/{self.credentials['account_id']}/orders", json_body=payload)

    def positions(self) -> Iterable[Dict[str, Any]]:
        data = self._request("GET", f"/accounts/{self.credentials['account_id']}/openPositions")
        return data.get("positions", [])

    def account_snapshot(self) -> Dict[str, Any]:
        return self._request("GET", f"/accounts/{self.credentials['account_id']}")


__all__ = ["OANDAConnector"]
