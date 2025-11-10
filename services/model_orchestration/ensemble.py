"""Model blending utilities relying solely on the standard library."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Deque, Dict, Iterable, Mapping, MutableMapping


@dataclass
class ModelPerformance:
    returns: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    accuracy: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def update(self, new_returns: Iterable[float], new_accuracy: float) -> None:
        self.returns.extend(new_returns)
        self.accuracy.append(new_accuracy)

    def rolling_sharpe(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        if len(set(self.returns)) == 1:
            return 0.0
        return mean(self.returns) / pstdev(self.returns)

    def rolling_accuracy(self) -> float:
        if not self.accuracy:
            return 0.0
        return mean(self.accuracy)


class EnsembleController:
    def __init__(self, window: int = 50):
        self.window = window
        self.models: MutableMapping[str, ModelPerformance] = {}

    def register_model(self, name: str) -> None:
        if name not in self.models:
            self.models[name] = ModelPerformance(
                returns=deque(maxlen=self.window),
                accuracy=deque(maxlen=self.window),
            )

    def record_performance(self, name: str, returns: Iterable[float], accuracy: float) -> None:
        self.register_model(name)
        self.models[name].update(returns, accuracy)

    def compute_weights(self) -> Dict[str, float]:
        if not self.models:
            return {}
        scores: Dict[str, float] = {}
        for name, perf in self.models.items():
            sharpe = max(perf.rolling_sharpe(), 0.0)
            accuracy = max(perf.rolling_accuracy(), 0.0)
            scores[name] = sharpe + accuracy
        total = sum(scores.values())
        if total == 0:
            equal = 1 / len(scores)
            return {name: equal for name in scores}
        return {name: score / total for name, score in scores.items()}

    def blend(self, predictions: Mapping[str, float]) -> float:
        if not predictions:
            raise ValueError("No predictions supplied")
        weights = self.compute_weights()
        if not weights:
            equal = 1 / len(predictions)
            weights = {name: equal for name in predictions}
        return sum(predictions[name] * weights.get(name, 0.0) for name in predictions)


__all__ = ["ModelPerformance", "EnsembleController"]
