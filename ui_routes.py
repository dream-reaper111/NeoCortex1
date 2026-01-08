"""UI routes shared across NeoCortex FastAPI apps."""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from auth import get_current_user
from fastapi import HTTPException


def _is_dev_mode() -> bool:
    env = os.getenv("NEOCORTEX_ENV", "").lower()
    debug = os.getenv("NEOCORTEX_DEBUG", "").lower()
    return env in {"dev", "development"} or debug in {"1", "true", "yes"}


def build_ui_router(templates: Jinja2Templates) -> APIRouter:
    router = APIRouter()

    async def _maybe_user(request: Request):
        try:
            return await get_current_user(request)
        except HTTPException:
            return None

    @router.get("/login", name="login_page")
    async def login_page(request: Request):
        user = await _maybe_user(request)
        return templates.TemplateResponse("login.html", {"request": request, "user": user})

    @router.get("/admin/login")
    async def admin_login_page(request: Request):
        user = await _maybe_user(request)
        return templates.TemplateResponse("admin_login.html", {"request": request, "user": user})

    @router.get("/enduserapp")
    async def enduserapp_page(request: Request):
        user = await _maybe_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=302)
        return templates.TemplateResponse("enduserapp.html", {"request": request, "user": user})

    @router.get("/liquidity")
    async def liquidity_page(request: Request):
        user = await _maybe_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=302)
        return templates.TemplateResponse("radar.html", {"request": request, "user": user})

    @router.get("/radar")
    async def radar_page(request: Request):
        user = await _maybe_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=302)
        return templates.TemplateResponse("radar.html", {"request": request, "user": user})

    @router.get("/dashboard")
    async def dashboard_page(request: Request):
        user = await _maybe_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=302)
        return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

    @router.get("/admin")
    async def admin_page(request: Request):
        user = await _maybe_user(request)
        roles = {str(role).strip().lower() for role in (user or {}).get("roles", []) if role is not None}
        if not user or "admin" not in roles:
            return RedirectResponse(url="/login?error=not_admin", status_code=302)
        return templates.TemplateResponse("admin.html", {"request": request, "user": user})

    return router


def register_routes_diagnostics(app, *, enabled: bool | None = None) -> None:
    if enabled is None:
        enabled = _is_dev_mode()
    if not enabled:
        return
    if any(getattr(route, "path", None) == "/__routes" for route in app.router.routes):
        return

    @app.get("/__routes", response_class=PlainTextResponse)
    def list_routes() -> str:
        lines: Iterable[str] = (
            f\"{','.join(sorted(route.methods or []))} {route.path}\"
            for route in app.router.routes
            if hasattr(route, \"path\")
        )
        return \"\\n\".join(sorted(lines))


def register_app_diagnostics(app, *, module_file: str, enabled: bool | None = None) -> None:
    if enabled is None:
        enabled = _is_dev_mode()
    if not enabled:
        return
    if not any(getattr(route, \"path\", None) == \"/__whoami\" for route in app.router.routes):

        @app.get(\"/__whoami\")
        def whoami() -> Dict[str, str]:
            return {
                \"app_id\": str(id(app)),
                \"module_file\": module_file,
                \"cwd\": os.getcwd(),
                \"python\": sys.executable,
            }

    if not any(getattr(route, \"path\", None) == \"/__routes\" for route in app.router.routes):

        @app.get(\"/__routes\", response_class=PlainTextResponse)
        def list_routes() -> str:
            lines: Iterable[str] = (
                f\"{','.join(sorted(route.methods or []))} {route.path}\"
                for route in app.router.routes
                if hasattr(route, \"path\")
            )
            return \"\\n\".join(sorted(lines))


__all__ = [\"build_ui_router\", \"register_app_diagnostics\", \"register_routes_diagnostics\"]
