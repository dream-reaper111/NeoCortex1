"""ICT-style signal pipeline combining stochastic and regime models.

This module translates market microstructure heuristics (GBM drift/vol,
autocorrelation, GARCH volatility, OU mean reversion, tail risk, logistic
regimes, and a unified price step) into a single payload that can be sent to
automation backends.

Key equations (discrete-time friendly representations):

- Geometric Brownian Motion (GBM): S_{t+1} = S_t * exp( (μ - 0.5 σ²) Δt + σ √Δt * ε_t )
- Autocorrelation: ρ(k) = Cov(r_t, r_{t-k}) / Var(r_t)
- GARCH(1,1): σ_t² = α₀ + α₁ ε_{t-1}² + β₁ σ_{t-1}²
- Ornstein–Uhlenbeck (OU): x_{t+1} = x_t + θ (μ_X − x_t) Δt + σ √Δt * ε_t
- Power-law tails (Hill estimator): α̂ = k / Σ_{i=1..k} ln( r_(i) / r_(k) ), r_(i) sorted |r|
- Logistic map: x_{n+1} = r x_n (1 − x_n)
- Unified price step: ΔP = α M_t + β R_t + γ σ_t
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

EQUATIONS: Dict[str, str] = {
    "gbm": "S_{t+1} = S_t * exp((μ - 0.5 σ^2) Δt + σ √Δt * ε_t)",
    "autocorrelation": "ρ(k) = Cov(r_t, r_{t-k}) / Var(r_t)",
    "garch": "σ_t^2 = α_0 + α_1 ε_{t-1}^2 + β_1 σ_{t-1}^2",
    "ou": "x_{t+1} = x_t + θ (μ_X − x_t) Δt + σ √Δt * ε_t",
    "power_tail": "α̂ = k / Σ_{i=1..k} ln(r_(i) / r_(k)), r_(i) sorted |r|",
    "logistic_map": "x_{n+1} = r x_n (1 − x_n)",
    "unified": "ΔP = α M_t + β R_t + γ σ_t",
}


@dataclass
class ICTBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ICTPipelineConfig:
    autocorr_lags: Sequence[int] = (1, 3, 5)
    garch_alpha0: float = 1e-6
    garch_alpha1: float = 0.08
    garch_beta1: float = 0.9
    ou_theta: float = 0.35
    ou_dt: float = 1.0
    tail_fraction: float = 0.1
    momentum_window: int = 10
    reversion_window: int = 20
    vol_window: int = 20
    unified_weights: Tuple[float, float, float] = (1.0, 1.0, 0.5)


def parse_bars(raw: Iterable[Mapping[str, object]]) -> List[ICTBar]:
    bars: List[ICTBar] = []
    for idx, item in enumerate(raw):
        try:
            bars.append(
                ICTBar(
                    timestamp=_parse_ts(item["timestamp"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item.get("volume", 0.0)),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive guard for malformed payloads
            raise ValueError(f"Invalid bar at index {idx}: {exc}") from exc
    if not bars:
        raise ValueError("At least one bar is required")
    return bars


def build_ict_payload(
    bars: Sequence[ICTBar],
    config: ICTPipelineConfig | None = None,
    include_equations: bool = False,
) -> Dict[str, object]:
    cfg = config or ICTPipelineConfig()
    closes = [bar.close for bar in bars]
    returns = _log_returns(closes)
    gbm = _gbm_estimates(returns)
    autocorr = {f"lag_{lag}": _autocorrelation(returns[1:], lag) for lag in cfg.autocorr_lags}
    garch = _garch_forecast(returns[1:], cfg)
    ou = _ou_projection(closes, cfg)
    tail = _tail_exponent(returns[1:], cfg)
    logistic = _logistic_regime(garch["sigma"], autocorr.get("lag_1", 0.0))
    unified = _unified_step(returns[1:], ou, garch["sigma"], cfg)
    payload: Dict[str, object] = {
        "gbm": gbm,
        "autocorrelation": autocorr,
        "garch": garch,
        "ou": ou,
        "power_tail": tail,
        "logistic_map": logistic,
        "unified": unified,
    }
    if include_equations:
        payload["equations"] = EQUATIONS
    return payload


def _parse_ts(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported timestamp type: {type(value)}")


def _log_returns(closes: Sequence[float]) -> List[float]:
    returns: List[float] = [0.0]
    for prev, current in zip(closes, closes[1:]):
        if prev <= 0:
            returns.append(0.0)
        else:
            returns.append(math.log(current / prev))
    return returns


def _gbm_estimates(returns: Sequence[float]) -> Dict[str, float]:
    drift = mean(returns) if returns else 0.0
    vol = pstdev(returns) if len(returns) > 1 else abs(drift)
    return {
        "drift": drift,
        "vol": vol,
    }


def _autocorrelation(series: Sequence[float], lag: int) -> float:
    if lag <= 0 or len(series) <= lag:
        return 0.0
    mu = mean(series)
    numerator = sum((series[i] - mu) * (series[i - lag] - mu) for i in range(lag, len(series)))
    denominator = sum((x - mu) ** 2 for x in series)
    return 0.0 if denominator == 0 else numerator / denominator


def _garch_forecast(returns: Sequence[float], cfg: ICTPipelineConfig) -> Dict[str, float]:
    if not returns:
        return {"variance": 0.0, "sigma": 0.0}
    mu = mean(returns)
    variance = pstdev(returns) ** 2 if len(returns) > 1 else (returns[-1] - mu) ** 2
    for ret in returns[-200:]:
        eps = ret - mu
        variance = cfg.garch_alpha0 + cfg.garch_alpha1 * (eps**2) + cfg.garch_beta1 * variance
    variance = max(variance, 0.0)
    return {"variance": variance, "sigma": math.sqrt(variance)}


def _ou_projection(closes: Sequence[float], cfg: ICTPipelineConfig) -> Dict[str, float]:
    mean_level = mean(closes)
    current = closes[-1]
    deviation = current - mean_level
    expected = current + cfg.ou_theta * (mean_level - current) * cfg.ou_dt
    return {
        "mean_level": mean_level,
        "deviation": deviation,
        "expected_reversion": expected,
        "theta": cfg.ou_theta,
    }


def _tail_exponent(returns: Sequence[float], cfg: ICTPipelineConfig) -> Dict[str, float]:
    magnitudes = sorted((abs(r) for r in returns if r != 0.0), reverse=True)
    if not magnitudes:
        return {"alpha": 0.0, "sample": 0}
    k = max(1, int(len(magnitudes) * cfg.tail_fraction))
    tail = magnitudes[:k]
    floor = tail[-1]
    if floor <= 0:
        return {"alpha": 0.0, "sample": k}
    hill_sum = sum(math.log(value / floor) for value in tail)
    alpha = (k / hill_sum) if hill_sum > 0 else 0.0
    return {"alpha": alpha, "sample": k}


def _logistic_regime(sigma: float, autocorr: float) -> Dict[str, float]:
    x = max(0.01, min(0.99, sigma / (1.0 + sigma)))
    r = 3.5 + min(0.49, max(-0.49, autocorr))  # keep r inside a chaotic-but-stable band
    x_next = r * x * (1 - x)
    return {"r": r, "x": x, "x_next": x_next}


def _unified_step(returns: Sequence[float], ou: Dict[str, float], sigma: float, cfg: ICTPipelineConfig) -> Dict[str, float]:
    momentum_window = max(1, cfg.momentum_window)
    reversion_window = max(1, cfg.reversion_window)
    vol_window = max(1, cfg.vol_window)
    momentum = mean(returns[-momentum_window:]) if returns else 0.0
    reversion = -mean(returns[-reversion_window:]) + (-ou.get("deviation", 0.0))
    vol_component = sigma if len(returns) >= vol_window else sigma * 0.5
    weight_m, weight_r, weight_v = cfg.unified_weights
    delta = (weight_m * momentum) + (weight_r * reversion) + (weight_v * vol_component)
    return {
        "momentum": momentum,
        "reversion": reversion,
        "vol_component": vol_component,
        "weights": {
            "alpha": weight_m,
            "beta": weight_r,
            "gamma": weight_v,
        },
        "delta_price": delta,
    }


def coerce_config(raw: Mapping[str, object] | None) -> ICTPipelineConfig:
    if not raw:
        return ICTPipelineConfig()
    allowed = {field.name for field in fields(ICTPipelineConfig)}
    filtered = {key: raw[key] for key in raw if key in allowed}
    return ICTPipelineConfig(**filtered)


__all__ = [
    "ICTBar",
    "ICTPipelineConfig",
    "parse_bars",
    "build_ict_payload",
    "coerce_config",
    "EQUATIONS",
]
