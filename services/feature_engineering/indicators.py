"""Indicator computations operating on simple Python data structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence

from services.data_pipeline.ingest import OHLCVBar, SentimentBar


def _ema(values: Sequence[float], span: int) -> List[float]:
    alpha = 2 / (span + 1)
    ema_values: List[float] = []
    last: Optional[float] = None
    for value in values:
        if last is None:
            last = value
        else:
            last = alpha * value + (1 - alpha) * last
        ema_values.append(last)
    return ema_values


def _rolling_mean(values: Sequence[float], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for idx in range(len(values)):
        if idx + 1 < window:
            out.append(None)
        else:
            segment = values[idx + 1 - window : idx + 1]
            out.append(mean(segment))
    return out


def _rolling_std(values: Sequence[float], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for idx in range(len(values)):
        if idx + 1 < window:
            out.append(None)
        else:
            segment = values[idx + 1 - window : idx + 1]
            if len(set(segment)) == 1:
                out.append(0.0)
            else:
                out.append(pstdev(segment))
    return out


@dataclass
class IndicatorConfig:
    atr_window: int = 14
    rsi_window: int = 14
    short_ema: int = 12
    long_ema: int = 26
    delta_z_window: int = 20


@dataclass
class IndicatorRow:
    timestamp: datetime
    values: Dict[str, float]


def average_true_range(bars: Sequence[OHLCVBar], window: int) -> List[Optional[float]]:
    atr: List[Optional[float]] = []
    prev_close: Optional[float] = None
    trs: List[float] = []
    for bar in bars:
        true_ranges = [bar.high - bar.low]
        if prev_close is not None:
            true_ranges.append(abs(bar.high - prev_close))
            true_ranges.append(abs(bar.low - prev_close))
        tr = max(true_ranges)
        trs.append(tr)
        if len(trs) < window:
            atr.append(None)
        else:
            segment = trs[-window:]
            atr.append(sum(segment) / len(segment))
        prev_close = bar.close
    return atr


def relative_strength_index(bars: Sequence[OHLCVBar], window: int) -> List[Optional[float]]:
    gains: List[float] = []
    losses: List[float] = []
    rsi_values: List[Optional[float]] = [None]
    for prev, current in zip(bars, bars[1:]):
        change = current.close - prev.close
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
        if len(gains) < window:
            rsi_values.append(None)
            continue
        avg_gain = _ema(gains, window)[-1]
        avg_loss = _ema(losses, window)[-1]
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    return rsi_values


def ema_cross(bars: Sequence[OHLCVBar], short_window: int, long_window: int) -> Dict[str, List[Optional[float]]]:
    closes = [bar.close for bar in bars]
    short = _ema(closes, short_window)
    long = _ema(closes, long_window)
    signal: List[float] = []
    prev_direction: Optional[int] = None
    for s_val, l_val in zip(short, long):
        direction = 1 if s_val > l_val else 0
        if prev_direction is None:
            signal.append(0.0)
        else:
            signal.append(float(direction - prev_direction))
        prev_direction = direction
    return {
        f"ema_{short_window}": short,
        f"ema_{long_window}": long,
        "ema_cross_signal": signal,
    }


def delta_z_score(values: Sequence[float], window: int) -> List[Optional[float]]:
    deltas: List[Optional[float]] = []
    means = _rolling_mean(values, window)
    stds = _rolling_std(values, window)
    z_scores: List[Optional[float]] = []
    for value, mean_val, std_val in zip(values, means, stds):
        if mean_val is None or std_val in (None, 0):
            z_scores.append(None)
        else:
            z_scores.append((value - mean_val) / std_val if std_val else None)
    previous: Optional[float] = None
    for score in z_scores:
        if score is None or previous is None:
            deltas.append(None)
        else:
            deltas.append(score - previous)
        previous = score
    return deltas


def enrich_indicators(
    bars: Sequence[OHLCVBar],
    config: Optional[IndicatorConfig] = None,
    sentiment: Optional[Sequence[SentimentBar]] = None,
) -> List[Dict[str, float]]:
    config = config or IndicatorConfig()
    atr_values = average_true_range(bars, config.atr_window)
    rsi_values = relative_strength_index(bars, config.rsi_window)
    ema_values = ema_cross(bars, config.short_ema, config.long_ema)
    closes = [bar.close for bar in bars]
    delta_z_values = delta_z_score(closes, config.delta_z_window)
    sentiment_map = {bar.timestamp: bar.sentiment for bar in sentiment or []}

    enriched: List[Dict[str, float]] = []
    for idx, bar in enumerate(bars):
        row = {
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        row["atr"] = atr_values[idx] if atr_values[idx] is not None else 0.0
        row["rsi"] = rsi_values[idx] if rsi_values[idx] is not None else 50.0
        row["ema_short"] = ema_values[f"ema_{config.short_ema}"][idx]
        row["ema_long"] = ema_values[f"ema_{config.long_ema}"][idx]
        row["ema_cross_signal"] = ema_values["ema_cross_signal"][idx]
        row["delta_z"] = delta_z_values[idx] if delta_z_values[idx] is not None else 0.0
        row["sentiment"] = sentiment_map.get(bar.timestamp, 0.0)
        enriched.append(row)
    return enriched


def batch_enrich(
    frames: Dict[str, Sequence[OHLCVBar]],
    config: Optional[IndicatorConfig] = None,
    sentiment: Optional[Dict[str, Sequence[SentimentBar]]] = None,
) -> Dict[str, List[Dict[str, float]]]:
    enriched: Dict[str, List[Dict[str, float]]] = {}
    for ticker, bars in frames.items():
        enriched[ticker] = enrich_indicators(bars, config, (sentiment or {}).get(ticker))
    return enriched


__all__ = [
    "IndicatorConfig",
    "IndicatorRow",
    "average_true_range",
    "relative_strength_index",
    "ema_cross",
    "delta_z_score",
    "enrich_indicators",
    "batch_enrich",
]
