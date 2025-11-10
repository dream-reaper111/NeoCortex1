"""TradeLocker WebSocket connector."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import websocket

from .base import BrokerConnector

_TRADELOCKER_WS = "wss://stream.tradelocker.com"


@dataclass
class _SocketState:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    connection: websocket.WebSocketApp | None = None


class TradeLockerConnector(BrokerConnector):
    name = "tradelocker"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self._state = _SocketState()

    @classmethod
    def required_keys(cls) -> Iterable[str]:
        return ["token"]

    def _ensure_socket(self) -> None:
        if self._state.connection is not None:
            return

        headers = {"Authorization": f"Bearer {self.credentials['token']}"}

        def on_message(_, message: str) -> None:
            self._state.messages.append(json.loads(message))

        ws = websocket.WebSocketApp(
            _TRADELOCKER_WS,
            header=[f"{key}: {value}" for key, value in headers.items()],
            on_message=on_message,
        )
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        self._state.connection = ws

    def place_order(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        self._ensure_socket()
        message = {
            "type": "order",
            "symbol": symbol,
            "side": side.lower(),
            "quantity": quantity,
        }
        assert self._state.connection
        self._state.connection.send(json.dumps(message))
        return message

    def positions(self) -> Iterable[Dict[str, Any]]:
        return [msg for msg in self._state.messages if msg.get("type") == "position"]

    def account_snapshot(self) -> Dict[str, Any]:
        for msg in reversed(self._state.messages):
            if msg.get("type") == "account":
                return msg
        return {}


__all__ = ["TradeLockerConnector"]
