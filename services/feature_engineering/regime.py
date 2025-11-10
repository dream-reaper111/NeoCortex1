"""Regime classification on indicator dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence


def _rolling(values: Sequence[float], window: int) -> List[List[float]]:
    buffers: List[List[float]] = []
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        buffers.append(list(values[start : idx + 1]))
    return buffers


def _safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    if len(set(values)) == 1:
        return 0.0
    return pstdev(values)


@dataclass
class RegimeConfig:
    trend_threshold: float = 0.75
    range_threshold: float = 0.002
    squeeze_quantile: float = 0.2
    lookback: int = 20
    sentiment_spike: float = 1.5


def rolling_sharpe(returns: Sequence[float], window: int) -> List[float]:
    buffers = _rolling(returns, window)
    sharpe_values: List[float] = []
    for buf in buffers:
        if len(buf) < window:
            sharpe_values.append(0.0)
            continue
        mu = mean(buf)
        sigma = _safe_std(buf)
        sharpe_values.append(0.0 if sigma == 0 else mu / sigma)
    return sharpe_values


def label_regimes(rows: Sequence[Dict[str, float]], config: Optional[RegimeConfig] = None) -> List[Dict[str, float]]:
    config = config or RegimeConfig()
    if not rows:
        return []
    closes = [row["close"] for row in rows]
    returns = [0.0]
    for prev, current in zip(closes, closes[1:]):
        returns.append((current - prev) / prev if prev else 0.0)
    sharpe_series = rolling_sharpe(returns, config.lookback)
    atr_values = [row.get("atr", 0.0) for row in rows]
    sorted_atr = []
    for idx, atr in enumerate(atr_values):
        start = max(0, idx + 1 - config.lookback)
        window_slice = sorted(atr_values[start : idx + 1])
        if not window_slice:
            sorted_atr.append(atr)
        else:
            quantile_index = int((len(window_slice) - 1) * config.squeeze_quantile)
            sorted_atr.append(window_slice[quantile_index])

    labelled: List[Dict[str, float]] = []
    for idx, row in enumerate(rows):
        label = "neutral"
        ema_signal = row.get("ema_cross_signal", 0.0)
        sharpe_val = sharpe_series[idx]
        ret = returns[idx]
        atr = atr_values[idx]
        atr_q = sorted_atr[idx]
        sentiment = row.get("sentiment", 0.0)
        if ema_signal > 0 and sharpe_val > config.trend_threshold:
            label = "trend_up"
        elif ema_signal < 0 and sharpe_val < -config.trend_threshold:
            label = "trend_down"
        elif abs(ret) < config.range_threshold:
            label = "range"
        elif atr <= atr_q:
            label = "squeeze"
        if abs(sentiment) > config.sentiment_spike:
            label = "news_spike"
        enriched = dict(row)
        enriched["regime"] = label
        enriched["sharpe"] = sharpe_val
        labelled.append(enriched)
    return labelled


def batch_label(frames: Dict[str, Sequence[Dict[str, float]]], config: Optional[RegimeConfig] = None) -> Dict[str, List[Dict[str, float]]]:
    labelled: Dict[str, List[Dict[str, float]]] = {}
    for ticker, rows in frames.items():
        labelled[ticker] = label_regimes(rows, config)
    return labelled


__all__ = ["RegimeConfig", "rolling_sharpe", "label_regimes", "batch_label"]
