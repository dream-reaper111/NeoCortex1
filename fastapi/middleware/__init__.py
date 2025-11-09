"""Lightweight middleware shims used when the real FastAPI package is unavailable."""

from __future__ import annotations

from .httpsredirect import HTTPSRedirectMiddleware

__all__ = ["HTTPSRedirectMiddleware"]

