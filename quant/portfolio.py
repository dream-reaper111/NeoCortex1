"""Portfolio management toolkit for NeoCortex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass
class AccountSnapshot:
    broker: str
    equity: float
    cash: float
    positions: Dict[str, float]


class MultiBrokerSynchronizer:
    """Maintains a consolidated view of accounts across brokers."""

    def __init__(self) -> None:
        self._accounts: Dict[str, AccountSnapshot] = {}

    def upsert(self, snapshot: AccountSnapshot) -> None:
        self._accounts[snapshot.broker] = snapshot

    def remove(self, broker: str) -> None:
        self._accounts.pop(broker, None)

    def total_equity(self) -> float:
        return sum(acc.equity for acc in self._accounts.values())

    def consolidated_positions(self) -> Dict[str, float]:
        inventory: Dict[str, float] = {}
        for acc in self._accounts.values():
            for symbol, qty in acc.positions.items():
                inventory[symbol] = inventory.get(symbol, 0.0) + qty
        return inventory


class DynamicAllocator:
    """Allocate capital between strategies using Sharpe/win-rate feedback."""

    def allocate(self, performance: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for name, metrics in performance.items():
            sharpe = metrics.get("sharpe", 0.0)
            win_rate = metrics.get("win_rate", 0.0)
            trades = metrics.get("trades", 1)
            score = max(0.0, 0.7 * sharpe + 0.3 * win_rate) * (1 + trades / 100)
            scores[name] = score
        total = sum(scores.values()) or 1.0
        return {name: score / total for name, score in scores.items()}


@dataclass
class EquityCurveTargetResult:
    exposure_multiplier: float
    target_equity: float


class EquityCurveTarget:
    """Scales exposure to maintain a smooth equity trajectory."""

    def __init__(self, smoothing: float = 0.1) -> None:
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        self.smoothing = smoothing
        self._equity_target: float | None = None

    def update(self, equity: float) -> EquityCurveTargetResult:
        if self._equity_target is None:
            self._equity_target = equity
        self._equity_target = (1 - self.smoothing) * self._equity_target + self.smoothing * equity
        if self._equity_target == 0:
            multiplier = 1.0
        else:
            deviation = equity / self._equity_target
            multiplier = max(0.2, min(3.0, deviation))
        return EquityCurveTargetResult(exposure_multiplier=multiplier, target_equity=self._equity_target)
