"""Dashboard application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

try:  # pragma: no cover - optional dependency for lightweight test environments
    from starlette.middleware.cors import CORSMiddleware
except ModuleNotFoundError:  # pragma: no cover
    class CORSMiddleware:  # type: ignore
        """Fallback no-op CORS middleware when Starlette is unavailable."""

        def __init__(self, app, **_: object) -> None:
            self.app = app

        async def __call__(self, scope, receive, send):  # pragma: no cover - pass-through behaviour
            await self.app(scope, receive, send)


def create_app() -> FastAPI:
    from .api import router as dashboard_router
    from .frontend import build_dashboard_app

    app = FastAPI(title="NeoCortex Analytics Dashboard")
    app.include_router(dashboard_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    dash_app = build_dashboard_app(app)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_index():
        return dash_app.index()

    return app


__all__ = ["create_app"]
