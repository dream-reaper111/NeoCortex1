"""Dependency providers shared across the dashboard API and UI."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from services.analytics import TradingDataStore


@lru_cache(maxsize=1)
def get_data_store(source_path: Optional[str] = None) -> TradingDataStore:
    if source_path:
        return TradingDataStore(source_path=Path(source_path))
    return TradingDataStore()


__all__ = ["get_data_store"]
