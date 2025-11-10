"""Analytics helpers for dashboards and reporting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass
class ReplayFrame:
    timestamp: datetime
    price: float
    position: float
    pnl: float


class TradeReplayer:
    """Generates candle-by-candle snapshots of trade performance."""

    def replay(
        self,
        prices: Sequence[Tuple[datetime, float]],
        trades: Sequence[Tuple[datetime, float]],
    ) -> List[ReplayFrame]:
        sorted_prices = sorted(prices, key=lambda item: item[0])
        sorted_trades = sorted(trades, key=lambda item: item[0])
        trade_iter = iter(sorted_trades)
        next_trade = next(trade_iter, None)
        position = 0.0
        entry_price = 0.0
        frames: List[ReplayFrame] = []
        for timestamp, price in sorted_prices:
            while next_trade is not None and next_trade[0] <= timestamp:
                _, size = next_trade
                if position == 0:
                    entry_price = price
                position += size
                if position == 0:
                    entry_price = 0.0
                next_trade = next(trade_iter, None)
            pnl = (price - entry_price) * position if position else 0.0
            frames.append(ReplayFrame(timestamp=timestamp, price=price, position=position, pnl=pnl))
        return frames


class ProfitLossHeatmap:
    """Aggregates PnL by weekday and hour combinations."""

    def build(self, trades: Iterable[Tuple[datetime, float]]) -> Dict[str, Dict[int, float]]:
        heatmap: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for timestamp, pnl in trades:
            day = timestamp.strftime("%a")
            heatmap[day][timestamp.hour] += pnl
        return {day: dict(hours) for day, hours in heatmap.items()}


class RiskCorrelationMatrix:
    """Computes a correlation matrix from rolling returns."""

    def compute(self, returns: Mapping[str, Sequence[float]]) -> Dict[str, Dict[str, float]]:
        symbols = list(returns)
        matrix: Dict[str, Dict[str, float]] = {symbol: {} for symbol in symbols}
        for i, sym_a in enumerate(symbols):
            series_a = returns[sym_a]
            mean_a = mean(series_a)
            variance_a = sum((x - mean_a) ** 2 for x in series_a)
            for sym_b in symbols[i:]:
                series_b = returns[sym_b]
                mean_b = mean(series_b)
                variance_b = sum((x - mean_b) ** 2 for x in series_b)
                covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(series_a, series_b))
                denom = sqrt(variance_a * variance_b)
                corr = 0.0 if denom == 0 else covariance / denom
                matrix[sym_a][sym_b] = corr
                matrix[sym_b][sym_a] = corr
        return matrix


class EquityWaterfall:
    """Breaks down cumulative performance by strategy."""

    def build(self, contributions: Mapping[str, Sequence[float]]) -> List[Dict[str, float]]:
        steps: List[Dict[str, float]] = []
        cumulative = 0.0
        for strategy, values in contributions.items():
            delta = sum(values)
            cumulative += delta
            steps.append({"strategy": strategy, "delta": delta, "cumulative": cumulative})
        return steps


class AIExplanationLayer:
    """Produces lightweight explanations for model decisions."""

    def explain(
        self,
        signals: Mapping[str, float],
        rationale: Mapping[str, Tuple[str, float]],
    ) -> List[Dict[str, object]]:
        explanations: List[Dict[str, object]] = []
        for feature, score in signals.items():
            reason, weight = rationale.get(feature, ("unknown", 0.0))
            explanations.append({"feature": feature, "score": score, "reason": reason, "weight": weight})
        return sorted(explanations, key=lambda item: abs(item["score"] * item["weight"]), reverse=True)


__all__ = [
    "ReplayFrame",
    "TradeReplayer",
    "ProfitLossHeatmap",
    "RiskCorrelationMatrix",
    "EquityWaterfall",
    "AIExplanationLayer",
]
