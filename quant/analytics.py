"""Analytics and visualisation helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass
class ReplayFrame:
    timestamp: datetime
    price: float
    position: float
    pnl: float


class TradeReplayer:
    """Generates candle-by-candle snapshots of trade performance."""

    def replay(self, prices: Sequence[Tuple[datetime, float]], trades: Sequence[Tuple[datetime, float]]) -> List[ReplayFrame]:
        trade_iter = iter(sorted(trades, key=lambda t: t[0]))
        pending = next(trade_iter, None)
        position = 0.0
        entry_price = 0.0
        frames: List[ReplayFrame] = []
        for timestamp, price in sorted(prices, key=lambda p: p[0]):
            while pending is not None and pending[0] <= timestamp:
                _, size = pending
                if position + size == 0:
                    entry_price = 0.0
                elif position == 0:
                    entry_price = price
                position += size
                pending = next(trade_iter, None)
            pnl = (price - entry_price) * position if position else 0.0
            frames.append(ReplayFrame(timestamp=timestamp, price=price, position=position, pnl=pnl))
        return frames


class ProfitLossHeatmap:
    """Aggregates PnL by weekday/hour combinations."""

    def build(self, trades: Sequence[Tuple[datetime, float]]) -> Dict[str, Dict[int, float]]:
        heatmap: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for timestamp, pnl in trades:
            weekday = timestamp.strftime("%a")
            hour = timestamp.hour
            heatmap[weekday][hour] += pnl
        return {day: dict(hours) for day, hours in heatmap.items()}


class RiskCorrelationMatrix:
    """Computes a correlation matrix from rolling returns."""

    def compute(self, returns: Mapping[str, Sequence[float]]) -> Dict[str, Dict[str, float]]:
        assets = list(returns)
        matrix: Dict[str, Dict[str, float]] = {asset: {} for asset in assets}
        for i, asset_a in enumerate(assets):
            series_a = returns[asset_a]
            mean_a = sum(series_a) / len(series_a)
            for asset_b in assets[i:]:
                series_b = returns[asset_b]
                mean_b = sum(series_b) / len(series_b)
                cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(series_a, series_b)) / max(len(series_a) - 1, 1)
                std_a = math.sqrt(sum((a - mean_a) ** 2 for a in series_a) / max(len(series_a) - 1, 1))
                std_b = math.sqrt(sum((b - mean_b) ** 2 for b in series_b) / max(len(series_b) - 1, 1))
                if std_a == 0 or std_b == 0:
                    corr = 0.0
                else:
                    corr = cov / (std_a * std_b)
                matrix[asset_a][asset_b] = corr
                matrix[asset_b][asset_a] = corr
        return matrix


class EquityWaterfall:
    """Breaks down cumulative performance by strategy."""

    def build(self, contributions: Mapping[str, Sequence[float]]) -> List[Dict[str, float]]:
        steps: List[Dict[str, float]] = []
        running_total = 0.0
        for strategy, values in contributions.items():
            delta = sum(values)
            running_total += delta
            steps.append({"strategy": strategy, "delta": delta, "cumulative": running_total})
        return steps


class AIExplanationLayer:
    """Produces lightweight explanations for model decisions."""

    def explain(self, signals: Mapping[str, float], rationale: Mapping[str, Tuple[str, float]]) -> List[Dict[str, object]]:
        explanations: List[Dict[str, object]] = []
        for feature, score in signals.items():
            reason, weight = rationale.get(feature, ("unknown", 0.0))
            explanations.append({"feature": feature, "score": score, "reason": reason, "weight": weight})
        return sorted(explanations, key=lambda item: abs(item["score"] * item["weight"]), reverse=True)
