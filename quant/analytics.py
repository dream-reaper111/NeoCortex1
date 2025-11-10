"""Analytics helpers for dashboards and reporting."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class TradeReplay:
    """Generate candle-by-candle replay data for executed trades."""

    def playback(self, candles: Sequence[Dict[str, float]], trades: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
        markers = {trade["timestamp"]: trade for trade in trades}
        replay = []
        for candle in candles:
            entry = dict(candle)
            trade = markers.get(candle.get("timestamp"))
            if trade:
                entry["trade_price"] = trade.get("price")
                entry["trade_size"] = trade.get("size")
            replay.append(entry)
        return replay


@dataclass
class ProfitHeatmap:
    """Aggregate profit by weekday and hour for heatmap visualisation."""

    def build(self, pnl_events: Iterable[Tuple[datetime, float]]) -> Dict[str, Dict[int, float]]:
        heatmap: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for timestamp, pnl in pnl_events:
            weekday = timestamp.strftime("%a")
            heatmap[weekday][timestamp.hour] += pnl
        return {day: dict(hours) for day, hours in heatmap.items()}


@dataclass
class RiskCorrelationMatrix:
    """Compute rolling correlation matrix for asset returns."""

    def compute(self, returns: Dict[str, Sequence[float]]) -> Dict[Tuple[str, str], float]:
        symbols = list(returns)
        matrix: Dict[Tuple[str, str], float] = {}
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i:]:
                series_a = np.array(returns[sym_a])
                series_b = np.array(returns[sym_b])
                if len(series_a) < 2 or len(series_b) < 2:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(series_a, series_b)[0, 1])
                matrix[(sym_a, sym_b)] = corr
                matrix[(sym_b, sym_a)] = corr
        return matrix


@dataclass
class EquityWaterfall:
    """Break down equity curve contribution per strategy."""

    def compose(self, strategy_pnl: Dict[str, Sequence[float]]) -> List[Dict[str, float]]:
        output: List[Dict[str, float]] = []
        cumulative = 0.0
        for strategy, pnl_series in strategy_pnl.items():
            contribution = sum(pnl_series)
            cumulative += contribution
            output.append({"strategy": strategy, "contribution": contribution, "cumulative": cumulative})
        return output


@dataclass
class AIExplanationLayer:
    """Explain why a model generated a signal."""

    def explain(self, features: Dict[str, float], thresholds: Dict[str, Tuple[float, float]]) -> List[str]:
        explanations: List[str] = []
        for name, value in features.items():
            lower, upper = thresholds.get(name, (None, None))
            if lower is not None and value < lower:
                explanations.append(f"{name} below threshold: {value:.2f} < {lower:.2f}")
            elif upper is not None and value > upper:
                explanations.append(f"{name} above threshold: {value:.2f} > {upper:.2f}")
            else:
                explanations.append(f"{name} within range at {value:.2f}")
        return explanations
