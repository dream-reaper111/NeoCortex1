"""Economic calendar feed using generic REST providers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import requests


class EconomicCalendarFeed:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def fetch(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        response = requests.get(
            self.endpoint,
            params={"start": start.isoformat(), "end": end.isoformat()},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()


__all__ = ["EconomicCalendarFeed"]
