"""Binance spot trading connector."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, Iterable

import requests

from .base import BrokerConnector

_BINANCE_API = "https://api.binance.com"


class BinanceConnector(BrokerConnector):
    name = "binance"

    @classmethod
    def required_keys(cls) -> Iterable[str]:
        return ["api_key", "api_secret"]

    def _signed_request(self, method: str, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        query = "&".join(f"{key}={value}" for key, value in params.items())
        signature = hmac.new(
            self.credentials["api_secret"].encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        headers = {"X-MBX-APIKEY": self.credentials["api_key"]}
        response = requests.request(
            method,
            f"{_BINANCE_API}{path}",
            params={**params, "signature": signature},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def place_order(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        payload = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
        }
        return self._signed_request("POST", "/api/v3/order", payload)

    def positions(self) -> Iterable[Dict[str, Any]]:
        account = self._signed_request("GET", "/api/v3/account")
        return [balance for balance in account.get("balances", []) if float(balance.get("free", 0))]

    def account_snapshot(self) -> Dict[str, Any]:
        return self._signed_request("GET", "/api/v3/account")


__all__ = ["BinanceConnector"]
