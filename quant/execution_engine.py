"""Execution utilities such as smart order routing and latency tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SmartOrderRouter:
    """Slice orders across venues using a simple volume-proportional rule."""

    venue_liquidity: Dict[str, float]

    def route(self, quantity: float) -> Dict[str, float]:
        total_liquidity = sum(self.venue_liquidity.values())
        if total_liquidity <= 0:
            return {venue: 0.0 for venue in self.venue_liquidity}
        return {
            venue: quantity * (liquidity / total_liquidity)
            for venue, liquidity in self.venue_liquidity.items()
        }


@dataclass
class OrderBookHeatmap:
    """Prepare order book snapshots for charting libraries such as Recharts."""

    depth: int = 10

    def build(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict[str, List[Dict[str, float]]]:
        bids_sorted = sorted(bids, key=lambda item: item[0], reverse=True)[: self.depth]
        asks_sorted = sorted(asks, key=lambda item: item[0])[: self.depth]

        def _build_side(levels: List[Tuple[float, float]]) -> List[Dict[str, float]]:
            cumulative = 0.0
            heatmap = []
            for price, size in levels:
                cumulative += size
                heatmap.append({"price": price, "size": size, "cumulative": cumulative})
            return heatmap

        return {"bids": _build_side(bids_sorted), "asks": _build_side(asks_sorted)}


@dataclass
class LatencyTracker:
    """Track call → fill → webhook timings."""

    buffer: List[Dict[str, float]] = field(default_factory=list)
    max_samples: int = 100

    def record(self, api_sent: float, broker_fill: float, webhook_received: float) -> None:
        measurement = {
            "api_to_fill": broker_fill - api_sent,
            "fill_to_webhook": webhook_received - broker_fill,
            "total": webhook_received - api_sent,
        }
        self.buffer.append(measurement)
        if len(self.buffer) > self.max_samples:
            self.buffer.pop(0)

    def snapshot(self) -> Dict[str, float]:
        if not self.buffer:
            return {"api_to_fill": 0.0, "fill_to_webhook": 0.0, "total": 0.0}
        totals = {"api_to_fill": 0.0, "fill_to_webhook": 0.0, "total": 0.0}
        for item in self.buffer:
            for key in totals:
                totals[key] += item[key]
        count = len(self.buffer)
        return {key: value / count for key, value in totals.items()}
