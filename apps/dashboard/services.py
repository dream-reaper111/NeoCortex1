"""Business logic powering the dashboard endpoints and UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from services.analytics import (
    TradeReplayEvent,
    TradingDataStore,
    calculate_all_metrics,
    compute_rolling_risk_matrix,
    risk_matrix_to_heatmap,
    simulate_monte_carlo_equity,
    simulation_to_payload,
)


@dataclass
class DashboardPayloads:
    pnl_breakdown: Dict[str, float]
    win_rate_heatmap: Dict[str, object]
    equity_curve: Dict[str, List]
    risk_matrix: Dict[str, object]
    metrics: Dict[str, float]
    monte_carlo: Dict[str, object]


class DashboardService:
    def __init__(self, store: TradingDataStore) -> None:
        self._store = store

    def get_trades(self) -> pd.DataFrame:
        return self._store.load_trades()

    def pnl_breakdown(self) -> Dict[str, float]:
        trades = self.get_trades()
        grouped = trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
        return grouped.to_dict()

    def win_rate_heatmap(self) -> Dict[str, object]:
        trades = self.get_trades()
        pivot = trades.pivot_table(
            index="strategy",
            columns="symbol",
            values="winning_trade",
            aggfunc="mean",
        ).fillna(0.0)
        return {"index": list(pivot.index), "columns": list(pivot.columns), "values": pivot.values.tolist()}

    def equity_curve(self) -> Dict[str, List]:
        equity = self._store.load_equity_curve()
        return {
            "timestamp": equity["timestamp"].astype(str).tolist(),
            "value": equity["value"].astype(float).tolist(),
        }

    def risk_matrix(self, window: int = 30) -> Dict[str, object]:
        trades = self.get_trades()
        price_frame = trades.pivot_table(index="timestamp", columns="symbol", values="price").ffill().bfill()
        risk_matrix = compute_rolling_risk_matrix(price_frame, window)
        return risk_matrix_to_heatmap(risk_matrix)

    def portfolio_metrics(self) -> Dict[str, float]:
        trades = self.get_trades()
        daily_returns = trades.groupby(trades["timestamp"].dt.date)["pnl"].sum() / trades.groupby(
            trades["timestamp"].dt.date
        )["quantity"].sum().replace(0, float("nan"))
        equity_curve = self._store.load_equity_curve()["value"].pct_change().fillna(0)
        metrics = calculate_all_metrics(trades, daily_returns.fillna(0), equity_curve)
        return metrics

    def monte_carlo(self, num_simulations: int = 50) -> Dict[str, object]:
        equity = self._store.load_equity_curve()
        returns = equity["value"].pct_change().fillna(0)
        simulation = simulate_monte_carlo_equity(
            returns=returns,
            starting_equity=float(equity["value"].iloc[0]),
            num_simulations=num_simulations,
        )
        return simulation_to_payload(simulation)

    def summary_payloads(self) -> DashboardPayloads:
        return DashboardPayloads(
            pnl_breakdown=self.pnl_breakdown(),
            win_rate_heatmap=self.win_rate_heatmap(),
            equity_curve=self.equity_curve(),
            risk_matrix=self.risk_matrix(),
            metrics=self.portfolio_metrics(),
            monte_carlo=self.monte_carlo(),
        )

    def trade_replay_events(self) -> List[TradeReplayEvent]:
        return list(self._store.iter_trade_replay())

    def trade_replay_payload(self, limit: int = 50) -> List[dict]:
        events = self.trade_replay_events()
        selected = events[-limit:]
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "symbol": event.symbol,
                "side": event.side,
                "quantity": event.quantity,
                "price": event.price,
                "pnl": event.pnl,
                "explain_text": event.explain_text,
            }
            for event in selected
        ]


__all__ = ["DashboardService", "DashboardPayloads"]
