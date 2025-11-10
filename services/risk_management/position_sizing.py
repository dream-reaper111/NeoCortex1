"""Risk-aware utilities implemented without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class SizingConfig:
    account_equity: float
    risk_per_trade: float = 0.01
    atr_multiplier: float = 1.5
    vix_floor: float = 12.0
    vix_ceiling: float = 40.0


def atr_position_size(atr: float, config: SizingConfig) -> float:
    if atr <= 0:
        raise ValueError("ATR must be positive")
    dollar_risk = config.account_equity * config.risk_per_trade
    stop_distance = atr * config.atr_multiplier
    return max(dollar_risk / stop_distance, 0.0)


def vix_volatility_scaler(vix_value: float, config: SizingConfig) -> float:
    span = config.vix_ceiling - config.vix_floor
    if span <= 0:
        return 1.0
    clamped = min(max(vix_value, config.vix_floor), config.vix_ceiling)
    scale = 1 - (clamped - config.vix_floor) / span
    return max(0.1, min(scale, 1.0))


def equity_curve_drawdown(equity_curve: Iterable[float]) -> List[float]:
    drawdowns: List[float] = []
    peak = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = 0.0 if peak == 0 else (equity - peak) / peak
        drawdowns.append(drawdown)
    return drawdowns


def is_max_drawdown_exceeded(equity_curve: Iterable[float], limit: float) -> bool:
    drawdowns = equity_curve_drawdown(equity_curve)
    return min(drawdowns) <= -abs(limit)


def adaptive_position_size(atr: float, vix_value: float, config: SizingConfig) -> float:
    base = atr_position_size(atr, config)
    scale = vix_volatility_scaler(vix_value, config)
    return base * scale


__all__ = [
    "SizingConfig",
    "atr_position_size",
    "vix_volatility_scaler",
    "equity_curve_drawdown",
    "is_max_drawdown_exceeded",
    "adaptive_position_size",
]
