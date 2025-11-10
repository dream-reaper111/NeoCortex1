"""Interactive Brokers connector using the Client Portal Web API."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import requests

from .base import BrokerConnector

_IBKR_API = "https://localhost:5000/v1/api"


class IBKRConnector(BrokerConnector):
    name = "ibkr"

    @classmethod
    def required_keys(cls) -> Iterable[str]:
        return ["base_url", "token"]

    def _request(self, method: str, path: str, json_body: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = f"{self.credentials.get('base_url', _IBKR_API)}{path}"
        headers = {"Authorization": f"Bearer {self.credentials['token']}"}
        response = requests.request(method, url, json=json_body, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        return response.json()

    def place_order(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        payload = {
            "conid": symbol,
            "orderType": "MKT",
            "side": side.upper(),
            "quantity": quantity,
        }
        return self._request("POST", "/iserver/account/orders", json_body=payload)

    def positions(self) -> Iterable[Dict[str, Any]]:
        data = self._request("GET", "/portfolio/positions")
        return data

    def account_snapshot(self) -> Dict[str, Any]:
        return self._request("GET", "/portfolio/accounts")


__all__ = ["IBKRConnector"]
