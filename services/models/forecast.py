"""PyTorch forecasters used for short-term predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

try:  # pragma: no cover - optional dependency guard
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class SequenceModelConfig:
    input_dim: int
    hidden_dim: int = 64
    output_dim: int = 1
    num_layers: int = 2
    dropout: float = 0.1
    sequence_length: int = 32
    device: str = "cpu"


if nn is not None:

    class PriceLSTMForecaster(nn.Module):
        """A lightweight LSTM model for predicting next step returns."""

        def __init__(self, config: SequenceModelConfig):
            super().__init__()
            self.config = config
            self.lstm = nn.LSTM(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(config.hidden_dim, config.output_dim)

        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            lstm_out, _ = self.lstm(x)
            last = lstm_out[:, -1, :]
            return self.head(last)


    class PriceTransformerForecaster(nn.Module):
        """Transformer encoder that ingests price features."""

        def __init__(self, config: SequenceModelConfig, nhead: int = 4):
            super().__init__()
            self.config = config
            self.positional = nn.Parameter(torch.zeros(1, config.sequence_length, config.input_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.input_dim,
                nhead=nhead,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.output_dim),
            )

        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            x = x + self.positional
            encoded = self.encoder(x)
            pooled = encoded.mean(dim=1)
            return self.head(pooled)

else:  # pragma: no cover

    class PriceLSTMForecaster:  # type: ignore[misc]
        def __init__(self, *args, **kwargs):  # noqa: D401
            raise RuntimeError("PyTorch is required for PriceLSTMForecaster")


    class PriceTransformerForecaster:  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for PriceTransformerForecaster")


class ForecastTrainer:
    """Generic mini-batch trainer used by both sequence models."""

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for ForecastTrainer")
        self.model = model
        self.device = next(model.parameters()).device
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, loader: Iterable[Tuple[Tensor, Tensor]]) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            preds = self.model(features)
            loss = self.criterion(preds, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += float(loss.detach())
            count += 1
        return total_loss / max(count, 1)

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Tuple[Tensor, Tensor]]) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            preds = self.model(features)
            loss = self.criterion(preds, targets)
            total_loss += float(loss)
            count += 1
        return total_loss / max(count, 1)


__all__ = [
    "SequenceModelConfig",
    "PriceLSTMForecaster",
    "PriceTransformerForecaster",
    "ForecastTrainer",
]
