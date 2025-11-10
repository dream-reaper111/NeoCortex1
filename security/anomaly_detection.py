"""Simple streaming anomaly detection for transactional activity."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Deque, Iterable, List


@dataclass
class TransactionAnomaly:
    amount: float
    zscore: float


@dataclass
class TransactionAnomalyDetector:
    window: int = 200
    threshold: float = 3.0
    _values: Deque[float] = field(default_factory=deque)

    def update(self, amounts: Iterable[float]) -> List[TransactionAnomaly]:
        anomalies: List[TransactionAnomaly] = []
        for amount in amounts:
            self._values.append(amount)
            if len(self._values) > self.window:
                self._values.popleft()
            if len(self._values) < 2:
                continue
            mu = mean(self._values)
            sigma = pstdev(self._values) or 1.0
            zscore = (amount - mu) / sigma
            if abs(zscore) >= self.threshold:
                anomalies.append(TransactionAnomaly(amount=amount, zscore=zscore))
        return anomalies


__all__ = ["TransactionAnomaly", "TransactionAnomalyDetector"]
