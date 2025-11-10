"""Base abstractions for broker connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class BrokerConnector(ABC):
    name: str

    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials

    @abstractmethod
    def place_order(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def positions(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def account_snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError

    def validate(self) -> None:
        required = self.required_keys()
        missing = [key for key in required if key not in self.credentials]
        if missing:
            raise ValueError(f"Missing credentials: {', '.join(missing)}")

    @classmethod
    def required_keys(cls) -> Iterable[str]:
        return []


__all__ = ["BrokerConnector"]
