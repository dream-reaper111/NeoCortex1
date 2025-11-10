"""Synthetic data ingestion primitives that do not rely on third-party libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, MutableMapping, Optional

import random


@dataclass
class OHLCVBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderFlowBar:
    timestamp: datetime
    bid_volume: float
    ask_volume: float

    @property
    def imbalance(self) -> float:
        total = self.bid_volume + self.ask_volume
        return 0.0 if total == 0 else (self.bid_volume - self.ask_volume) / total


@dataclass
class SentimentBar:
    timestamp: datetime
    sentiment: float
    buzz: float


@dataclass
class StructuredStore:
    ohlcv: MutableMapping[str, List[OHLCVBar]] = field(default_factory=dict)
    order_flow: MutableMapping[str, List[OrderFlowBar]] = field(default_factory=dict)
    sentiment: MutableMapping[str, List[SentimentBar]] = field(default_factory=dict)

    def snapshot(self, ticker: str) -> Dict[str, List]:
        return {
            "ohlcv": list(self.ohlcv.get(ticker, [])),
            "order_flow": list(self.order_flow.get(ticker, [])),
            "sentiment": list(self.sentiment.get(ticker, [])),
        }


@dataclass
class IngestionConfig:
    tickers: Iterable[str]
    start: datetime
    end: datetime
    frequency: timedelta = timedelta(minutes=60)
    seed: Optional[int] = None


class MarketDataIngestor:
    """Generates synthetic market data suitable for unit tests."""

    def __init__(self, config: IngestionConfig):
        if config.start >= config.end:
            raise ValueError("start must be before end")
        self.config = config
        self.rng = random.Random(config.seed)
        self.store = StructuredStore()

    def run(self) -> StructuredStore:
        for ticker in self.config.tickers:
            timeline = list(self._timeline())
            self.store.ohlcv[ticker] = self._build_ohlcv(timeline)
            self.store.order_flow[ticker] = self._build_order_flow(timeline)
            self.store.sentiment[ticker] = self._build_sentiment(timeline)
        return self.store

    # ------------------------------------------------------------------
    def _timeline(self) -> Iterable[datetime]:
        current = self.config.start.astimezone(timezone.utc)
        end = self.config.end.astimezone(timezone.utc)
        while current < end:
            yield current
            current += self.config.frequency

    def _build_ohlcv(self, timeline: List[datetime]) -> List[OHLCVBar]:
        series: List[OHLCVBar] = []
        price = 100.0 + self.rng.uniform(-1, 1)
        for stamp in timeline:
            drift = self.rng.gauss(0, 0.3)
            price = max(1.0, price + drift)
            high = price + abs(self.rng.gauss(0, 0.2))
            low = max(0.1, price - abs(self.rng.gauss(0, 0.2)))
            open_price = price + self.rng.gauss(0, 0.05)
            close_price = price + self.rng.gauss(0, 0.05)
            volume = max(100.0, 1000.0 + self.rng.gauss(0, 100))
            series.append(
                OHLCVBar(
                    timestamp=stamp,
                    open=open_price,
                    high=max(high, open_price, close_price),
                    low=min(low, open_price, close_price),
                    close=close_price,
                    volume=volume,
                )
            )
        return series

    def _build_order_flow(self, timeline: List[datetime]) -> List[OrderFlowBar]:
        series: List[OrderFlowBar] = []
        for stamp in timeline:
            bid = max(10.0, 500.0 + self.rng.gauss(0, 50))
            ask = max(10.0, 500.0 + self.rng.gauss(0, 50))
            series.append(OrderFlowBar(timestamp=stamp, bid_volume=bid, ask_volume=ask))
        return series

    def _build_sentiment(self, timeline: List[datetime]) -> List[SentimentBar]:
        series: List[SentimentBar] = []
        for stamp in timeline:
            sentiment = self.rng.gauss(0, 0.5)
            buzz = max(1.0, 50.0 + self.rng.gauss(0, 10))
            series.append(SentimentBar(timestamp=stamp, sentiment=sentiment, buzz=buzz))
        return series


__all__ = [
    "IngestionConfig",
    "MarketDataIngestor",
    "StructuredStore",
    "OHLCVBar",
    "OrderFlowBar",
    "SentimentBar",
]
