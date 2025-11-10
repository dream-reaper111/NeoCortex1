"""Strategy intelligence components for the NeoCortex quant terminal."""

from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Deque, Dict, List, Mapping, Sequence


ModelCallable = Callable[[Mapping[str, float]], float]


@dataclass
class EnsembleResult:
    """Outcome of an ensemble vote."""

    votes: Dict[str, float]
    consensus: str
    confidence: float


class AutoEnsembleEngine:
    """Run multiple models and aggregate their directional views."""

    def __init__(self, models: Mapping[str, ModelCallable], threshold: float = 0.0) -> None:
        if not models:
            raise ValueError("At least one model must be provided")
        self._models = dict(models)
        self.threshold = threshold

    def run(self, features: Mapping[str, float]) -> EnsembleResult:
        votes: Dict[str, float] = {}
        for name, model in self._models.items():
            score = float(model(features))
            votes[name] = score
        directional_votes = {
            name: ("long" if score > self.threshold else "short")
            for name, score in votes.items()
        }
        counts = Counter(directional_votes.values())
        consensus, count = counts.most_common(1)[0]
        confidence = count / len(directional_votes)
        return EnsembleResult(votes=votes, consensus=consensus, confidence=confidence)

    def update_model(self, name: str, model: ModelCallable) -> None:
        self._models[name] = model

    def remove_model(self, name: str) -> None:
        self._models.pop(name)


@dataclass
class RiskAdjustment:
    """Positioning recommendation returned by :class:`AdaptiveRiskModule`."""

    position_size: float
    take_profit: float
    stop_loss: float
    volatility: float


class AdaptiveRiskModule:
    """Simple risk engine that adapts exposure to realised volatility."""

    def __init__(
        self,
        target_volatility: float,
        base_position: float,
        lookback: int = 20,
        max_leverage: float = 3.0,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        self.target_volatility = target_volatility
        self.base_position = base_position
        self.lookback = lookback
        self.max_leverage = max_leverage
        self._returns: Deque[float] = deque(maxlen=lookback)

    def update(self, return_pct: float) -> RiskAdjustment:
        self._returns.append(return_pct)
        if len(self._returns) < 2:
            vol = abs(return_pct)
        else:
            vol = pstdev(self._returns) * math.sqrt(252)
        if vol == 0:
            leverage = self.max_leverage
        else:
            leverage = min(self.max_leverage, self.target_volatility / max(vol, 1e-8))
        position_size = self.base_position * leverage
        tp = return_pct + (vol * 0.5)
        sl = return_pct - (vol * 0.5)
        return RiskAdjustment(position_size=position_size, take_profit=tp, stop_loss=sl, volatility=vol)


class RegimeDetector:
    """Classifies market regimes from price series using simple heuristics."""

    def __init__(self, trend_threshold: float = 1.5, squeeze_threshold: float = 0.5, lookback: int = 50) -> None:
        if lookback < 5:
            raise ValueError("lookback must be at least 5")
        self.trend_threshold = trend_threshold
        self.squeeze_threshold = squeeze_threshold
        self.lookback = lookback

    def classify(self, prices: Sequence[float]) -> str:
        if len(prices) < self.lookback:
            raise ValueError("Insufficient data for regime detection")
        window = prices[-self.lookback :]
        returns = [math.log(window[i + 1] / window[i]) for i in range(len(window) - 1)]
        avg_return = mean(returns)
        vol = pstdev(returns) * math.sqrt(252)
        sharpe = avg_return / max(pstdev(returns) if len(returns) > 1 else 1e-6, 1e-6)
        if abs(sharpe) > self.trend_threshold:
            return "trend"
        if vol < self.squeeze_threshold:
            return "squeeze"
        return "chop"


class MonteCarloStressTester:
    """Run Monte-Carlo analysis on trade sequences."""

    def __init__(self, simulations: int = 1000) -> None:
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        self.simulations = simulations

    def simulate(self, trade_returns: Sequence[float], sequence_length: int | None = None) -> Dict[str, float]:
        if not trade_returns:
            raise ValueError("trade_returns must not be empty")
        length = sequence_length or len(trade_returns)
        drawdowns: List[float] = []
        equity_curves: List[List[float]] = []
        for _ in range(self.simulations):
            cumulative = 1.0
            peak = 1.0
            curve = [cumulative]
            for ret in random.choices(trade_returns, k=length):
                cumulative *= 1 + ret
                peak = max(peak, cumulative)
                dd = (cumulative - peak) / peak
                drawdowns.append(dd)
                curve.append(cumulative)
            equity_curves.append(curve)
        worst_dd = min(drawdowns)
        avg_dd = sum(drawdowns) / len(drawdowns)
        tail = sorted(drawdowns)[: max(1, len(drawdowns) // 20)]
        return {
            "worst_drawdown": worst_dd,
            "average_drawdown": avg_dd,
            "tail_risk": sum(tail) / len(tail),
            "expected_equity": sum(curve[-1] for curve in equity_curves) / len(equity_curves),
        }
