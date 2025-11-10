"""Lightweight middleware shims used when the real FastAPI package is unavailable."""
from __future__ import annotations

from .httpsredirect import HTTPSRedirectMiddleware
from .cors import CORSMiddleware

__all__ = ["HTTPSRedirectMiddleware", "CORSMiddleware"]
