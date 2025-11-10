"""Strategy intelligence utilities for orchestrating AI-driven trading models."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Sequence, Tuple


@dataclass
class AutoEnsembleEngine:
    """Combine heterogeneous model opinions into a single trading decision."""

    weights: Dict[str, float] = field(default_factory=dict)

    def vote(self, signals: Dict[str, float]) -> float:
        """Return the weighted consensus signal.

        Positive numbers indicate a long bias, negative numbers a short bias. The
        magnitude is normalised to ``[-1, 1]`` so callers can map it to position
        sizing logic.
        """

        if not signals:
            return 0.0

        if not self.weights:
            self.weights = {name: 1.0 for name in signals}

        total_weight = sum(self.weights.get(name, 0.0) for name in signals)
        if total_weight == 0:
            return 0.0

        weighted_signal = sum(
            self.weights.get(name, 0.0) * value for name, value in signals.items()
        )
        consensus = weighted_signal / total_weight
        return max(-1.0, min(1.0, consensus))


@dataclass
class AdaptiveRiskModule:
    """Derive dynamic position sizes and risk parameters from volatility."""

    base_position: float = 1.0
    min_multiplier: float = 0.25
    max_multiplier: float = 2.0

    def compute(self, volatility_score: float) -> Tuple[float, float, float]:
        """Return position size, take-profit, and stop-loss multipliers.

        ``volatility_score`` should be normalised to ``[0, 1]``.
        """

        volatility_score = max(0.0, min(1.0, volatility_score))
        multiplier = self.max_multiplier - (self.max_multiplier - self.min_multiplier) * volatility_score
        position_size = self.base_position * multiplier

        # Higher volatility implies wider stops/targets.
        tp_multiplier = 1.0 + volatility_score
        sl_multiplier = 1.0 + (1.0 - volatility_score)
        return position_size, tp_multiplier, sl_multiplier


@dataclass
class RegimeDetector:
    """Classify market regimes using lightweight heuristics."""

    trend_threshold: float = 0.4
    squeeze_threshold: float = 0.1

    def classify(self, returns: Sequence[float]) -> str:
        if not returns:
            return "unknown"

        avg_return = mean(returns)
        volatility = (mean(abs(r) for r in returns) or 1e-9)
        signal_to_noise = abs(avg_return) / volatility

        if signal_to_noise > self.trend_threshold:
            return "trend"
        if volatility < self.squeeze_threshold:
            return "squeeze"
        return "chop"


@dataclass
class MonteCarloStressTester:
    """Perform Monte-Carlo stress scenarios over historical trades."""

    random_state: random.Random = field(default_factory=random.Random)

    def simulate(
        self,
        trade_returns: Sequence[float],
        num_simulations: int = 1000,
        path_length: int | None = None,
    ) -> Dict[str, float]:
        if not trade_returns:
            return {"max_drawdown": 0.0, "var_95": 0.0, "cvar_95": 0.0}

        path_length = path_length or len(trade_returns)
        max_drawdowns: List[float] = []
        ending_balances: List[float] = []

        for _ in range(num_simulations):
            sampled = [self.random_state.choice(trade_returns) for _ in range(path_length)]
            equity = 1.0
            peak = 1.0
            max_dd = 0.0
            for r in sampled:
                equity *= (1 + r)
                peak = max(peak, equity)
                drawdown = (peak - equity) / peak
                max_dd = max(max_dd, drawdown)
            max_drawdowns.append(max_dd)
            ending_balances.append(equity)

        sorted_balances = sorted(ending_balances)
        var_index = max(0, int(0.05 * len(sorted_balances)) - 1)
        var_95 = sorted_balances[var_index] - 1.0
        cvar_95 = mean(sorted_balances[: var_index + 1]) - 1.0 if var_index >= 0 else 0.0

        return {
            "max_drawdown": max_drawdowns and max(max_drawdowns) or 0.0,
            "median_drawdown": max_drawdowns and sorted(max_drawdowns)[len(max_drawdowns) // 2] or 0.0,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }
