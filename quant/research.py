"""Research and machine-learning utilities."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


class ModelTrainer:
    """Dispatches to TensorFlow or PyTorch for lightweight model fine-tuning."""

    def __init__(self) -> None:
        self._tf = self._load_optional("tensorflow")
        self._torch = self._load_optional("torch")

    def _load_optional(self, module: str) -> Any | None:
        if importlib.util.find_spec(module) is None:
            return None
        return importlib.import_module(module)

    def train(self, data: Sequence[Tuple[List[float], float]], framework: str = "torch", epochs: int = 1) -> Dict[str, Any]:
        if framework == "tensorflow":
            if self._tf is None:
                raise RuntimeError("TensorFlow is not installed")
            model = self._tf.keras.Sequential([
                self._tf.keras.layers.InputLayer(input_shape=(len(data[0][0]),)),
                self._tf.keras.layers.Dense(8, activation="relu"),
                self._tf.keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            x = [features for features, _ in data]
            y = [target for _, target in data]
            history = model.fit(x, y, epochs=epochs, verbose=0)
            return {"framework": "tensorflow", "loss": history.history["loss"][-1]}
        if self._torch is None:
            raise RuntimeError("PyTorch is not installed")
        torch = self._torch
        input_dim = len(data[0][0])
        model = torch.nn.Sequential(torch.nn.Linear(input_dim, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        for _ in range(epochs):
            for features, target in data:
                x = torch.tensor(features, dtype=torch.float32)
                y = torch.tensor([target], dtype=torch.float32)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
        return {"framework": "torch", "loss": float(loss.detach().cpu().item())}


class FeatureLab:
    """Automatically engineers a set of market microstructure features."""

    def generate(self, prices: Sequence[float], volumes: Sequence[float]) -> Dict[str, float]:
        if len(prices) != len(volumes):
            raise ValueError("prices and volumes must align")
        if not prices:
            raise ValueError("prices must not be empty")
        mean_price = sum(prices) / len(prices)
        mean_volume = sum(volumes) / len(volumes)
        volume_delta = volumes[-1] - volumes[0]
        price_delta = prices[-1] - prices[0]
        ob_imbalance = (volumes[-1] - mean_volume) / max(mean_volume, 1e-6)
        volatility = sum(abs(p - mean_price) for p in prices) / len(prices)
        return {
            "mean_price": mean_price,
            "mean_volume": mean_volume,
            "volume_delta": volume_delta,
            "price_delta": price_delta,
            "orderbook_imbalance": ob_imbalance,
            "volatility": volatility,
        }


@dataclass
class ModelVersion:
    name: str
    version: int
    metrics: Dict[str, float]


class ModelRegistry:
    """Keeps track of model versions and benchmarks."""

    def __init__(self) -> None:
        self._models: Dict[str, List[ModelVersion]] = {}

    def register(self, name: str, metrics: Dict[str, float]) -> ModelVersion:
        version = len(self._models.get(name, [])) + 1
        model_version = ModelVersion(name=name, version=version, metrics=metrics)
        self._models.setdefault(name, []).append(model_version)
        return model_version

    def best(self, name: str, metric: str = "sharpe") -> ModelVersion:
        versions = self._models.get(name)
        if not versions:
            raise KeyError(name)
        return max(versions, key=lambda mv: mv.metrics.get(metric, float("-inf")))


class ReinforcementLearner:
    """Simple reinforcement learning agent using tabular Q-learning."""

    def __init__(self, actions: Sequence[str], learning_rate: float = 0.1, discount: float = 0.95) -> None:
        self.actions = list(actions)
        self.learning_rate = learning_rate
        self.discount = discount
        self._q_table: Dict[Tuple[int, ...], Dict[str, float]] = {}

    def _state_key(self, state: Sequence[int]) -> Tuple[int, ...]:
        return tuple(state)

    def policy(self, state: Sequence[int]) -> str:
        key = self._state_key(state)
        values = self._q_table.get(key, {})
        if not values:
            return self.actions[0]
        return max(values.items(), key=lambda item: item[1])[0]

    def update(self, state: Sequence[int], action: str, reward: float, next_state: Sequence[int]) -> None:
        key = self._state_key(state)
        q_values = self._q_table.setdefault(key, {a: 0.0 for a in self.actions})
        next_key = self._state_key(next_state)
        next_values = self._q_table.get(next_key, {a: 0.0 for a in self.actions})
        best_next = max(next_values.values()) if next_values else 0.0
        q_values[action] += self.learning_rate * (reward + self.discount * best_next - q_values[action])


class ExplainableAI:
    """Produces per-feature contribution scores using a simple gradient proxy."""

    def attribute(self, inputs: Sequence[float], weights: Sequence[float]) -> List[float]:
        if len(inputs) != len(weights):
            raise ValueError("inputs and weights must be the same length")
        baseline = sum(weights) / len(weights)
        return [(x - baseline) * w for x, w in zip(inputs, weights)]
