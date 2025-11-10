"""Centralized data access layer for dashboard analytics.

This module exposes :class:`TradingDataStore` which encapsulates the logic for
retrieving normalized trade/equity/model signal data that is shared across the
API endpoints and the dashboard UI.  A lightweight in-memory store is used by
default so that the analytics layer can function out of the box, but the class
is designed so it can be backed by a database or data lake in production.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
import random

import pandas as pd

try:  # pragma: no cover - optional dependency for stochastic sampling
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


@dataclass(frozen=True)
class TradeReplayEvent:
    """Dataclass representing a single event in the trade replay stream."""

    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    pnl: float
    explain_text: str


class TradingDataStore:
    """Provides a standardized interface for querying trading data.

    Parameters
    ----------
    source_path:
        Optional path to a parquet/csv file containing historical trades.  When
        omitted the store will synthesize a deterministic yet realistic looking
        data set which is perfectly adequate for testing and local development.
    seed:
        Optional random seed used when generating sample data.
    """

    def __init__(self, source_path: Optional[Path] = None, seed: Optional[int] = None) -> None:
        self._source_path = Path(source_path) if source_path else None
        self._seed = seed or 13
        self._trades_cache: Optional[pd.DataFrame] = None
        self._equity_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_sample_trades(self, periods: int = 250) -> pd.DataFrame:
        if np is not None:
            rng = np.random.default_rng(self._seed)
            normal = rng.normal
            choice = rng.choice
            integer = lambda low, high: rng.integers(low, high)
        else:  # pragma: no cover - deterministic fallback when numpy is unavailable
            rng = random.Random(self._seed)
            normal = lambda loc, scale: rng.normalvariate(loc, scale)
            choice = lambda options: rng.choice(list(options))
            integer = lambda low, high: rng.randrange(low, high)
        start = datetime.now() - timedelta(days=periods)
        timestamps = [start + timedelta(days=i) for i in range(periods)]
        symbols = ["EURUSD", "AAPL", "BTCUSD", "ES_F", "NQ_F"]
        strategies = ["mean_rev", "momentum", "breakout"]
        records: List[dict] = []
        equity = 1_000_000.0
        for ts in timestamps:
            symbol = choice(symbols)
            side = choice(["LONG", "SHORT"])
            qty = float(integer(1, 5)) * 10.0
            price = float(normal(100, 15))
            pnl = float(normal(2_500, 7_500))
            equity += pnl
            records.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "strategy": choice(strategies),
                    "side": side,
                    "quantity": qty,
                    "price": price,
                    "pnl": pnl,
                    "equity": equity,
                    "explain_text": f"Model rationale for {symbol} {side.lower()} at {price:.2f}.",
                }
            )
        trades = pd.DataFrame.from_records(records)
        trades["return"] = trades["pnl"] / trades["quantity"].replace(0, pd.NA)
        trades["winning_trade"] = trades["pnl"] > 0
        return trades

    def _load_from_disk(self) -> pd.DataFrame:
        if not self._source_path:
            raise FileNotFoundError("No source path configured for TradingDataStore")
        if not self._source_path.exists():
            raise FileNotFoundError(f"Trade source file not found: {self._source_path}")
        if self._source_path.suffix == ".parquet":
            trades = pd.read_parquet(self._source_path)
        else:
            trades = pd.read_csv(self._source_path, parse_dates=["timestamp"])
        trades = trades.sort_values("timestamp").reset_index(drop=True)
        if "return" not in trades:
            trades["return"] = trades["pnl"] / trades["quantity"].replace(0, pd.NA)
        if "winning_trade" not in trades:
            trades["winning_trade"] = trades["pnl"] > 0
        if "equity" not in trades:
            trades["equity"] = trades["pnl"].cumsum()
        if "explain_text" not in trades:
            trades["explain_text"] = "Model explanation unavailable"
        return trades

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_trades(self) -> pd.DataFrame:
        """Return the normalized trade history DataFrame."""

        if self._trades_cache is None:
            if self._source_path is None:
                self._trades_cache = self._generate_sample_trades()
            else:
                self._trades_cache = self._load_from_disk()
        return self._trades_cache.copy()

    def load_equity_curve(self) -> pd.DataFrame:
        """Return the equity curve derived from the trade history."""

        if self._equity_cache is None:
            trades = self.load_trades()
            equity = trades.loc[:, ["timestamp", "equity"]].copy()
            equity.rename(columns={"equity": "value"}, inplace=True)
            equity["value"] = equity["value"].astype(float)
            self._equity_cache = equity
        return self._equity_cache.copy()

    def iter_trade_replay(self) -> Iterator[TradeReplayEvent]:
        """Yield trades in chronological order for websocket streaming."""

        trades = self.load_trades()
        for record in trades.itertuples(index=False):
            yield TradeReplayEvent(
                timestamp=record.timestamp,
                symbol=record.symbol,
                side=record.side,
                quantity=float(record.quantity),
                price=float(record.price),
                pnl=float(record.pnl),
                explain_text=str(record.explain_text),
            )

    def get_symbols(self) -> List[str]:
        trades = self.load_trades()
        return sorted(trades["symbol"].unique())

    def get_strategies(self) -> List[str]:
        trades = self.load_trades()
        return sorted(trades["strategy"].unique())

    def rolling_returns(self, window: int = 20) -> pd.Series:
        trades = self.load_trades()
        daily_returns = trades.groupby(trades["timestamp"].dt.date)["pnl"].sum()
        return daily_returns.rolling(window=window, min_periods=1).mean()

    def fetch_trade_batches(self, batch_size: int = 20) -> Iterable[pd.DataFrame]:
        trades = self.load_trades()
        for start in range(0, len(trades), batch_size):
            yield trades.iloc[start : start + batch_size]


__all__ = ["TradingDataStore", "TradeReplayEvent"]
