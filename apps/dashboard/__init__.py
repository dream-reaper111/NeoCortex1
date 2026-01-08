"""Dashboard application factory."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

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
    templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[2] / "templates"))
    app.include_router(dashboard_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    dash_app = build_dashboard_app(app)

    @app.get("/dashboard")
    async def dashboard_index(request: Request):
        return templates.TemplateResponse("dashboard.html", {"request": request})

    return app


__all__ = ["create_app"]
