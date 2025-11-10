"""Utilities for computing rolling correlation/risk matrices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

try:  # pragma: no cover - optional dependency for matrix helpers
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


@dataclass
class RiskMatrixResult:
    labels: List[str]
    matrix: List[List[float]]
    window: int


def compute_rolling_risk_matrix(price_frame: pd.DataFrame, window: int = 30) -> RiskMatrixResult:
    """Compute a rolling correlation matrix for the provided price data."""

    if price_frame.empty:
        return RiskMatrixResult(labels=[], matrix=[], window=window)
    price_frame = price_frame.sort_index()
    returns = price_frame.pct_change().dropna()
    if returns.empty:
        return RiskMatrixResult(labels=list(price_frame.columns), matrix=[], window=window)
    rolling_corr = returns.rolling(window=window, min_periods=max(5, window // 2)).corr()

    last_timestamp = rolling_corr.index.get_level_values(0).max()
    latest_corr = rolling_corr.xs(last_timestamp, level=0)
    labels = list(price_frame.columns)
    size = len(labels)
    if np is not None:
        matrix = np.zeros((size, size), dtype=float)
    else:
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i, col_i in enumerate(labels):
        for j, col_j in enumerate(labels):
            value = 1.0 if col_i == col_j else float(latest_corr.get(col_j, {}).get(col_i, float("nan")))
            if np is not None:
                matrix[i, j] = value
            else:
                matrix[i][j] = value
    if np is not None:
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        matrix_list = matrix.tolist()
    else:
        matrix_list = [
            [
                0.0
                if value != value  # NaN check
                else max(min(value, 1.0), -1.0)
            for value in row
            ]
            for row in matrix
        ]
    return RiskMatrixResult(labels=labels, matrix=matrix_list, window=window)


def risk_matrix_to_heatmap(matrix: RiskMatrixResult) -> Dict[str, object]:
    """Convert a :class:`RiskMatrixResult` into a JSON serialisable payload."""

    return {
        "labels": matrix.labels,
        "matrix": matrix.matrix,
        "window": matrix.window,
    }


__all__ = ["RiskMatrixResult", "compute_rolling_risk_matrix", "risk_matrix_to_heatmap"]
