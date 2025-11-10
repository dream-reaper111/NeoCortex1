"""Minimal CORS middleware compatible with the FastAPI stub."""
from __future__ import annotations

import inspect
from typing import Awaitable, Callable, Iterable, Optional, Sequence, Set

from ..responses import Response
from .. import Request


class CORSMiddleware:
    """Very small subset of Starlette's CORS middleware behaviour.

    The goal of this stub is simply to support the handful of options used by
    ``server.py`` while keeping the implementation dependency free.  It mirrors
    the signature of the real middleware so application code does not need to
    change when the real FastAPI package is installed.
    """

    def __init__(
        self,
        app: Callable[..., Response],
        *,
        allow_origins: Optional[Sequence[str]] = None,
        allow_methods: Optional[Iterable[str]] = None,
        allow_headers: Optional[Iterable[str]] = None,
        allow_credentials: bool = False,
        expose_headers: Optional[Iterable[str]] = None,
        max_age: int = 600,
        allow_origin_regex: Optional[str] = None,
    ) -> None:
        self.app = app
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        self.allow_methods = self._normalise(allow_methods, default={"GET", "POST", "OPTIONS"})
        self.allow_headers = self._normalise(allow_headers, default={"*"})
        self.expose_headers = self._normalise(expose_headers, preserve_case=True)
        self.allow_all_origins = False
        origins = list(allow_origins or [])
        if "*" in origins:
            self.allow_all_origins = True
            origins = ["*"]
        self.allow_origins = set(origins)
        self.allow_origin_regex = allow_origin_regex

    @staticmethod
    def _normalise(
        values: Optional[Iterable[str]],
        default: Optional[Set[str]] = None,
        *,
        preserve_case: bool = False,
    ) -> Optional[Set[str]]:
        if values is None:
            return set(default) if default is not None else None
        if preserve_case:
            return {value for value in values if isinstance(value, str)}
        return {value.upper() if isinstance(value, str) else value for value in values}

    def _is_allowed_origin(self, origin: Optional[str]) -> bool:
        if origin is None:
            return False
        if self.allow_all_origins:
            return True
        if origin in self.allow_origins:
            return True
        if self.allow_origin_regex is not None:
            import re

            if re.match(self.allow_origin_regex, origin):
                return True
        return False

    def _configure_preflight_response(self, request: Request, origin: str) -> Response:
        request_method = request.headers.get("access-control-request-method", "").upper()
        if request_method and self.allow_methods is not None and "*" not in self.allow_methods:
            if request_method not in self.allow_methods:
                return Response(b"", status_code=400)
        headers = request.headers.get("access-control-request-headers", "")
        requested_headers = {header.strip().upper() for header in headers.split(",") if header.strip()}
        if requested_headers and self.allow_headers is not None and "*" not in self.allow_headers:
            if not requested_headers.issubset(self.allow_headers):
                return Response(b"", status_code=400)
        response = Response(b"", status_code=200)
        self._apply_cors_headers(response, origin, is_preflight=True)
        return response

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response] | Response],
    ) -> Response:
        origin = request.headers.get("origin")
        if request.method.upper() == "OPTIONS" and origin and self._is_allowed_origin(origin):
            return self._configure_preflight_response(request, origin)

        response = call_next(request)
        if inspect.isawaitable(response):
            response = await response
        if not isinstance(response, Response):
            response = Response(response)
        if origin and self._is_allowed_origin(origin):
            self._apply_cors_headers(response, origin, is_preflight=False)
        return response

    def _apply_cors_headers(self, response: Response, origin: str, *, is_preflight: bool) -> None:
        if self.allow_all_origins and not self.allow_credentials:
            response.headers["access-control-allow-origin"] = "*"
        else:
            response.headers["access-control-allow-origin"] = origin
        if self.allow_credentials:
            response.headers["access-control-allow-credentials"] = "true"
        if self.expose_headers:
            response.headers["access-control-expose-headers"] = ", ".join(sorted(self.expose_headers))
        if is_preflight:
            if self.allow_methods:
                if "*" in self.allow_methods:
                    response.headers["access-control-allow-methods"] = "*"
                else:
                    response.headers["access-control-allow-methods"] = ", ".join(sorted(self.allow_methods))
            if self.allow_headers:
                if "*" in self.allow_headers:
                    response.headers["access-control-allow-headers"] = "*"
                else:
                    response.headers["access-control-allow-headers"] = ", ".join(sorted(self.allow_headers))
            response.headers["access-control-max-age"] = str(self.max_age)

    def __call__(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response] | Response],
    ) -> Awaitable[Response]:
        return self.dispatch(request, call_next)
