"""Monte Carlo equity simulation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd
import random

try:  # pragma: no cover - optional dependency for vectorised sampling
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


@dataclass
class MonteCarloSimulation:
    paths: pd.DataFrame
    starting_equity: float
    num_simulations: int
    num_periods: int


def simulate_monte_carlo_equity(
    returns: Iterable[float],
    starting_equity: float,
    num_simulations: int = 100,
    num_periods: Optional[int] = None,
    seed: Optional[int] = None,
) -> MonteCarloSimulation:
    """Simulate Monte Carlo equity paths using bootstrapped daily returns."""

    returns_list = [float(r) for r in returns if r == r]
    if not returns_list:
        returns_list = [0.0]
    if num_periods is None:
        num_periods = len(returns_list)
    if np is not None:
        returns_array = np.array(returns_list, dtype=float)
        rng = np.random.default_rng(seed)
        samples = rng.choice(returns_array, size=(num_simulations, num_periods), replace=True)
        cumulative = np.cumprod(1 + samples, axis=1)
        equity_paths = starting_equity * cumulative
        index = pd.RangeIndex(start=1, stop=num_periods + 1, name="period")
        paths = pd.DataFrame(equity_paths.T, index=index)
    else:  # pragma: no cover - fallback when numpy is unavailable
        rng = random.Random(seed)
        path_columns = {}
        for sim in range(num_simulations):
            equity = []
            value = float(starting_equity)
            for ret in rng.choices(returns_list, k=num_periods):
                value *= 1 + ret
                equity.append(value)
            path_columns[sim] = equity
        index = pd.RangeIndex(start=1, stop=num_periods + 1, name="period")
        paths = pd.DataFrame(path_columns, index=index)
    return MonteCarloSimulation(
        paths=paths,
        starting_equity=float(starting_equity),
        num_simulations=int(num_simulations),
        num_periods=int(num_periods),
    )


def simulation_to_payload(simulation: MonteCarloSimulation) -> Dict[str, object]:
    return {
        "starting_equity": simulation.starting_equity,
        "num_simulations": simulation.num_simulations,
        "num_periods": simulation.num_periods,
        "paths": simulation.paths.to_dict(orient="list"),
    }


__all__ = ["MonteCarloSimulation", "simulate_monte_carlo_equity", "simulation_to_payload"]
