"""Analytics toolkit for portfolio dashboards."""
from .data_store import TradeReplayEvent, TradingDataStore
from .metrics import (
    calculate_all_metrics,
    calculate_calmar_ratio,
    calculate_expectancy,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from .monte_carlo import MonteCarloSimulation, simulate_monte_carlo_equity, simulation_to_payload
from .risk_matrix import RiskMatrixResult, compute_rolling_risk_matrix, risk_matrix_to_heatmap

__all__ = [
    "TradeReplayEvent",
    "TradingDataStore",
    "calculate_all_metrics",
    "calculate_calmar_ratio",
    "calculate_expectancy",
    "calculate_profit_factor",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "MonteCarloSimulation",
    "simulate_monte_carlo_equity",
    "simulation_to_payload",
    "RiskMatrixResult",
    "compute_rolling_risk_matrix",
    "risk_matrix_to_heatmap",
]
