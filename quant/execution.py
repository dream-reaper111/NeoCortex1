"""Execution engine primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class Venue:
    name: str
    max_volume: float
    fee_bps: float


@dataclass
class RoutedOrder:
    venue: str
    quantity: float
    fee: float


class SmartOrderRouter:
    """Naïve smart-order router that splits orders by venue capacity and fees."""

    def route(self, quantity: float, venues: Sequence[Venue]) -> List[RoutedOrder]:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if not venues:
            raise ValueError("venues must not be empty")
        remaining = quantity
        plan: List[RoutedOrder] = []
        sorted_venues = sorted(venues, key=lambda v: (v.fee_bps, -v.max_volume))
        for venue in sorted_venues:
            if remaining <= 0:
                break
            fill = min(venue.max_volume, remaining)
            if fill <= 0:
                continue
            plan.append(RoutedOrder(venue=venue.name, quantity=fill, fee=fill * venue.fee_bps / 10000))
            remaining -= fill
        if remaining > 1e-6:
            last = sorted_venues[-1]
            plan.append(RoutedOrder(venue=last.name, quantity=remaining, fee=remaining * last.fee_bps / 10000))
        return plan


@dataclass
class HeatmapCell:
    price: float
    liquidity: float


class OrderBookHeatmap:
    """Aggregates order-book levels into a heatmap representation."""

    def build(self, bids: Sequence[Tuple[float, float]], asks: Sequence[Tuple[float, float]], bin_size: float) -> Dict[str, List[HeatmapCell]]:
        if bin_size <= 0:
            raise ValueError("bin_size must be positive")
        heatmap = {"bids": self._bin_levels(bids, -abs(bin_size)), "asks": self._bin_levels(asks, abs(bin_size))}
        return heatmap

    def _bin_levels(self, levels: Sequence[Tuple[float, float]], bin_size: float) -> List[HeatmapCell]:
        buckets: Dict[float, float] = {}
        for price, size in levels:
            bucket = round(price / bin_size) * bin_size
            buckets[bucket] = buckets.get(bucket, 0.0) + size
        return [HeatmapCell(price=price, liquidity=liquidity) for price, liquidity in sorted(buckets.items())]


@dataclass
class LatencyEvent:
    name: str
    timestamp: float


@dataclass
class LatencyStats:
    total_latency_ms: float
    per_hop_ms: Dict[str, float]


class LatencyTracker:
    """Records latency between execution hops."""

    def __init__(self) -> None:
        self._events: List[LatencyEvent] = []

    def record(self, name: str, timestamp: float | None = None) -> None:
        self._events.append(LatencyEvent(name=name, timestamp=timestamp or time.time()))

    def compute(self) -> LatencyStats:
        if len(self._events) < 2:
            raise ValueError("Need at least two events to compute latency")
        sorted_events = sorted(self._events, key=lambda e: e.timestamp)
        per_hop: Dict[str, float] = {}
        for prev, current in zip(sorted_events, sorted_events[1:]):
            delta_ms = (current.timestamp - prev.timestamp) * 1000
            per_hop[f"{prev.name}->{current.name}"] = delta_ms
        total = sorted_events[-1].timestamp - sorted_events[0].timestamp
        return LatencyStats(total_latency_ms=total * 1000, per_hop_ms=per_hop)

    def reset(self) -> None:
        self._events.clear()
