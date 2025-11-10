"""Portfolio management utilities for orchestrating multi-strategy capital."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping


@dataclass
class BrokerAccount:
    """Represents a broker connection and basic portfolio stats."""

    name: str
    equity: float
    buying_power: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class MultiBrokerSync:
    """Combine broker snapshots into a consolidated book."""

    accounts: List[BrokerAccount] = field(default_factory=list)

    def merge(self) -> Dict[str, float]:
        total_equity = sum(account.equity for account in self.accounts)
        total_bp = sum(account.buying_power for account in self.accounts)
        return {"equity": total_equity, "buying_power": total_bp}


@dataclass
class DynamicAllocator:
    """Allocate capital between strategies based on rolling performance."""

    floor_weight: float = 0.05

    def rebalance(self, sharpe_ratios: Mapping[str, float], win_rates: Mapping[str, float]) -> Dict[str, float]:
        if not sharpe_ratios:
            return {}

        scores = {}
        for name, sharpe in sharpe_ratios.items():
            win_rate = win_rates.get(name, 0.5)
            scores[name] = max(self.floor_weight, sharpe * 0.7 + win_rate * 0.3)

        total = sum(scores.values())
        return {name: value / total for name, value in scores.items()} if total else {}


@dataclass
class EquityCurveTargeting:
    """Adjust exposure to maintain a desired equity glide-path."""

    target_growth: float = 0.01
    dampening: float = 0.5

    def scale(self, realised_return: float) -> float:
        performance_gap = self.target_growth - realised_return
        adjustment = 1.0 + performance_gap * self.dampening
        return max(0.0, adjustment)
