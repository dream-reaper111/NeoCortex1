"""UI routes shared across NeoCortex FastAPI apps."""
from __future__ import annotations

import os
from typing import Iterable

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates


def _is_dev_mode() -> bool:
    env = os.getenv("NEOCORTEX_ENV", "").lower()
    debug = os.getenv("NEOCORTEX_DEBUG", "").lower()
    return env in {"dev", "development"} or debug in {"1", "true", "yes"}


def build_ui_router(templates: Jinja2Templates) -> APIRouter:
    router = APIRouter()

    @router.get("/login", name="login_page")
    def login_page(request: Request):
        return templates.TemplateResponse("login.html", {"request": request})

    @router.get("/admin/login")
    def admin_login_page(request: Request):
        return templates.TemplateResponse("admin_login.html", {"request": request})

    @router.get("/enduserapp")
    def enduserapp_page(request: Request):
        return templates.TemplateResponse("enduserapp.html", {"request": request})

    @router.get("/liquidity")
    def liquidity_page(request: Request):
        return templates.TemplateResponse("radar.html", {"request": request})

    @router.get("/radar")
    def radar_page(request: Request):
        return templates.TemplateResponse("radar.html", {"request": request})

    @router.get("/dashboard")
    def dashboard_page(request: Request):
        return templates.TemplateResponse("dashboard.html", {"request": request})

    @router.get("/admin")
    def admin_page(request: Request):
        return templates.TemplateResponse("admin.html", {"request": request})

    return router


def register_routes_diagnostics(app, *, enabled: bool | None = None) -> None:
    if enabled is None:
        enabled = _is_dev_mode()
    if not enabled:
        return

    @app.get("/__routes", response_class=PlainTextResponse)
    def list_routes() -> str:
        lines: Iterable[str] = (
            f\"{','.join(sorted(route.methods or []))} {route.path}\"
            for route in app.router.routes
            if hasattr(route, \"path\")
        )
        return \"\\n\".join(sorted(lines))


__all__ = [\"build_ui_router\", \"register_routes_diagnostics\"]
