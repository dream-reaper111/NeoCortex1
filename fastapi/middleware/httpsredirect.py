"""Minimal HTTPS redirect middleware compatible with the FastAPI stub."""

from __future__ import annotations

from typing import Callable

from ..responses import RedirectResponse


class HTTPSRedirectMiddleware:
    """Redirect any plain-HTTP request to the HTTPS version of the URL."""

    def __init__(self, app: Callable[..., object], status_code: int = 307) -> None:
        self.app = app
        self.status_code = status_code

    async def dispatch(self, request, call_next):  # type: ignore[override]
        scheme = getattr(request.url, "scheme", request.headers.get("x-forwarded-proto", "http"))
        if scheme == "https":
            return await call_next(request)

        host = request.headers.get("host") or getattr(request.url, "netloc", "")
        path = request.path or "/"
        query = request.query
        if query:
            path = f"{path}?{query}"
        target = f"https://{host}{path}" if host else "https://" + path.lstrip("/")
        return RedirectResponse(url=target, status_code=self.status_code)

