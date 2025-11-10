"""AI research tooling for NeoCortex."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence


@dataclass
class TrainingEndpoint:
    """Wrap a callable trainer for use with a REST endpoint."""

    trainer: Callable[[Dict[str, Any]], Dict[str, Any]]

    def train(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.trainer(payload)


@dataclass
class FeatureLab:
    """Generate engineered features for models."""

    def build(self, candles: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
        features: List[Dict[str, float]] = []
        for i, candle in enumerate(candles):
            prev_close = candles[i - 1]["close"] if i else candle["close"]
            delta = candle["close"] - prev_close
            features.append(
                {
                    "close": candle["close"],
                    "volume_delta": candle.get("volume", 0.0) - candles[i - 1].get("volume", 0.0) if i else 0.0,
                    "ob_imbalance": candle.get("bid_volume", 0.0) - candle.get("ask_volume", 0.0),
                    "volatility_skew": abs(delta) / (candle.get("high", 1.0) - candle.get("low", 1.0) or 1.0),
                }
            )
        return features


@dataclass
class ModelRegistry:
    """Track model versions and performance metrics."""

    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def register(self, name: str, version: str, metrics: Dict[str, float]) -> None:
        self.registry[name] = {"version": version, "metrics": metrics}

    def best_model(self, metric: str = "sharpe") -> str | None:
        best_name = None
        best_score = float("-inf")
        for name, payload in self.registry.items():
            score = payload.get("metrics", {}).get(metric)
            if score is not None and score > best_score:
                best_name = name
                best_score = score
        return best_name


@dataclass
class ReinforcementLearningAgent:
    """Toy reinforcement learning agent using epsilon-greedy policy."""

    epsilon: float = 0.1
    q_values: Dict[str, float] = field(default_factory=dict)

    def select_action(self, actions: Sequence[str]) -> str:
        if not actions:
            raise ValueError("No actions provided")
        if random.random() < self.epsilon:
            return random.choice(list(actions))
        return max(actions, key=lambda action: self.q_values.get(action, 0.0))

    def update(self, action: str, reward: float, learning_rate: float = 0.1) -> None:
        current = self.q_values.get(action, 0.0)
        self.q_values[action] = current + learning_rate * (reward - current)


@dataclass
class ExplainableAI:
    """Placeholder for XAI hooks such as SHAP or Grad-CAM."""

    def describe(self, inputs: Dict[str, float], gradients: Dict[str, float]) -> List[str]:
        explanations: List[str] = []
        for feature, value in inputs.items():
            gradient = gradients.get(feature, 0.0)
            explanations.append(f"{feature}: contribution {gradient:.3f} at value {value:.3f}")
        return explanations
