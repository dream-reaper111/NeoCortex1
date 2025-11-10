"""Portfolio analytics helper functions."""
from __future__ import annotations

from typing import Dict, Iterable

import math
import pandas as pd

try:  # pragma: no cover - optional dependency for numerical operations
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


def _annualize_factor(periods_per_year: int) -> float:
    return float(periods_per_year)


def calculate_sharpe_ratio(returns: Iterable[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Compute the annualized Sharpe ratio."""

    series = pd.Series(list(returns)).dropna()
    if series.empty:
        return 0.0
    excess_returns = series - risk_free_rate / _annualize_factor(periods_per_year)
    std = excess_returns.std(ddof=1)
    if std == 0:
        return 0.0
    return (periods_per_year ** 0.5) * excess_returns.mean() / std


def calculate_sortino_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized Sortino ratio."""

    series = pd.Series(list(returns)).dropna()
    if series.empty:
        return 0.0
    excess_returns = series - risk_free_rate / _annualize_factor(periods_per_year)
    downside = excess_returns[excess_returns < 0]
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0
    return (periods_per_year ** 0.5) * excess_returns.mean() / downside_std


def calculate_calmar_ratio(equity_curve: Iterable[float], periods_per_year: int = 252) -> float:
    """Compute the Calmar ratio using drawdown of the equity curve."""

    equity = pd.Series(list(equity_curve)).dropna()
    if equity.empty:
        return 0.0
    cumulative_returns = equity.pct_change().fillna(0)
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0, float("nan"))
    max_drawdown = drawdown.min()
    if max_drawdown == 0 or (np.isnan(max_drawdown) if np is not None else math.isnan(float(max_drawdown))):
        return 0.0
    annual_return = cumulative_returns.mean() * periods_per_year
    return float(annual_return / abs(max_drawdown))


def calculate_expectancy(trades: pd.DataFrame) -> float:
    """Compute expectancy (average profit per trade)."""

    if trades.empty:
        return 0.0
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / len(trades) if len(trades) else 0.0
    loss_rate = 1 - win_rate
    avg_win = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = abs(losses["pnl"].mean()) if not losses.empty else 0.0
    return win_rate * avg_win - loss_rate * avg_loss


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """Compute the profit factor."""

    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_all_metrics(
    trades: pd.DataFrame,
    returns: Iterable[float],
    equity_curve: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Compute all portfolio metrics in a single call."""

    returns_series = pd.Series(list(returns)).dropna()
    metrics = {
        "sharpe": calculate_sharpe_ratio(returns_series, risk_free_rate, periods_per_year),
        "sortino": calculate_sortino_ratio(returns_series, risk_free_rate, periods_per_year),
        "calmar": calculate_calmar_ratio(list(equity_curve), periods_per_year),
        "expectancy": calculate_expectancy(trades),
        "profit_factor": calculate_profit_factor(trades),
    }
    return {key: float(value) for key, value in metrics.items()}


__all__ = [
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_expectancy",
    "calculate_profit_factor",
    "calculate_all_metrics",
]
