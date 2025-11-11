"""Reusable CSRF token management for NeoCortex FastAPI routes."""
from __future__ import annotations

import hmac

import os
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Optional, Set

from fastapi import HTTPException, Request, Response
from starlette.datastructures import FormData
from starlette.status import HTTP_403_FORBIDDEN

SAFE_METHODS: Set[str] = {"GET", "HEAD", "OPTIONS", "TRACE"}


def _bool_env(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


@dataclass
class _TokenState:
    token: str
    expires_at: datetime


class CSRFManager:
    """Manage CSRF session cookies and token validation."""

    def __init__(
        self,
        *,
        cookie_name: str = "nc_csrf_session",
        header_name: str = "x-csrf-token",
        field_name: str = "csrf_token",
        ttl: timedelta = timedelta(minutes=30),
        secure_cookie: bool = True,
        cookie_path: str = "/",
        cookie_domain: Optional[str] = None,
        exempt_paths: Optional[Iterable[str]] = None,
    ) -> None:
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.field_name = field_name
        self.ttl = ttl
        self.secure_cookie = secure_cookie
        self.cookie_path = cookie_path
        self.cookie_domain = cookie_domain
        self.samesite = "Strict"
        self._tokens: Dict[str, _TokenState] = {}
        self._lock = threading.Lock()
        self._exempt_paths: Set[str] = set(exempt_paths or ())

    def add_exempt_path(self, path: str) -> None:
        normalized = path.split("?")[0]
        self._exempt_paths.add(normalized)

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _purge_expired(self) -> None:
        now = self._now()
        expired: Set[str] = set()
        for session_id, state in self._tokens.items():
            if state.expires_at <= now:
                expired.add(session_id)
        if not expired:
            return
        for session_id in expired:
            self._tokens.pop(session_id, None)

    def _normalize_path(self, path: str) -> str:
        return path.split("?")[0]

    def _valid_session_id(self, value: str) -> bool:
        return bool(value) and len(value) <= 256

    def get_session_id(self, request: Request) -> Optional[str]:
        cookie = request.cookies.get(self.cookie_name)
        if cookie and self._valid_session_id(cookie):
            return cookie
        return None

    def ensure_cookie(self, request: Request, response: Response) -> str:
        session_id = self.get_session_id(request)
        if not session_id:
            session_id = secrets.token_urlsafe(32)
        # Refresh cookie expiry on each issuance
        response.set_cookie(
            self.cookie_name,
            session_id,
            httponly=True,
            secure=self.secure_cookie,
            samesite=self.samesite,
            max_age=int(self.ttl.total_seconds()),
            path=self.cookie_path,
            domain=self.cookie_domain,
        )
        return session_id

    def issue_token(self, request: Request, response: Response) -> str:
        session_id = self.ensure_cookie(request, response)
        token = secrets.token_hex(32)
        expires = self._now() + self.ttl
        with self._lock:
            self._purge_expired()
            self._tokens[session_id] = _TokenState(token=token, expires_at=expires)
        return token

    def validate_token(self, session_id: str, token: str) -> bool:
        with self._lock:
            self._purge_expired()
            state = self._tokens.get(session_id)
            if not state:
                return False
            if state.expires_at <= self._now():
                self._tokens.pop(session_id, None)
                return False
            if not hmac.compare_digest(state.token, token):
                return False
            # Sliding expiration window
            self._tokens[session_id] = _TokenState(
                token=state.token,
                expires_at=self._now() + self.ttl,
            )
            return True

    def clear(self, request: Request, response: Optional[Response] = None) -> None:
        session_id = self.get_session_id(request)
        if not session_id:
            return
        with self._lock:
            self._tokens.pop(session_id, None)
        if response is not None:
            response.delete_cookie(
                self.cookie_name,
                path=self.cookie_path,
                domain=self.cookie_domain,
                samesite=self.samesite,
                secure=self.secure_cookie,
            )

    async def verify_request(self, request: Request) -> None:
        if request.method.upper() in SAFE_METHODS:
            return
        path = self._normalize_path(str(request.url.path))
        if path in self._exempt_paths:
            return
        session_id = self.get_session_id(request)
        if not session_id:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="CSRF session missing")
        token = request.headers.get(self.header_name)
        if not token:
            content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
            if content_type in {"application/x-www-form-urlencoded", "multipart/form-data"}:
                form: Optional[FormData] = None
                try:
                    form = await request.form()
                except Exception:
                    form = None
                if form is not None:
                    token = form.get(self.field_name)  # type: ignore[arg-type]
        if not token or not isinstance(token, str):
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="CSRF token missing")
        if not self.validate_token(session_id, token):
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid CSRF token")

    async def __call__(self, request: Request) -> None:
        await self.verify_request(request)


def create_default_manager(*, secure_cookie: Optional[bool] = None) -> CSRFManager:
    secure = secure_cookie if secure_cookie is not None else _bool_env("CSRF_COOKIE_SECURE", True)
    manager = CSRFManager(secure_cookie=secure)
    manager.add_exempt_path("/csrf-token")
    return manager


csrf_manager = create_default_manager()


async def csrf_protect(request: Request) -> None:
    await csrf_manager.verify_request(request)


def issue_csrf_token(request: Request, response: Response) -> str:
    return csrf_manager.issue_token(request, response)


def clear_csrf(request: Request, response: Optional[Response] = None) -> None:
    csrf_manager.clear(request, response)
