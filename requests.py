"""Lightweight shim around :mod:`requests` for offline testing."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_third_party_requests():
    """Attempt to import the real ``requests`` package if available."""

    spec = importlib.util.find_spec("requests")
    if not spec or not spec.loader or not getattr(spec, "origin", None):
        return None

    origin_path = Path(spec.origin).resolve()
    this_path = Path(__file__).resolve()
    if origin_path == this_path:
        return None
    if origin_path.parent == this_path.parent / "__pycache__":
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_third_party_requests = _load_third_party_requests()

if _third_party_requests is not None:
    sys.modules[__name__] = _third_party_requests
    globals().update(_third_party_requests.__dict__)
else:
    import json
    import urllib.request
    from dataclasses import dataclass
    from typing import Any, Dict, Optional

    __all__ = [
        "Response",
        "HTTPError",
        "get",
        "post",
        "Session",
        "adapters",
        "exceptions",
    ]

    @dataclass
    class Response:
        status_code: int
        _body: bytes

        def json(self) -> Any:
            if not self._body:
                return {}
            return json.loads(self._body.decode("utf-8"))

        @property
        def text(self) -> str:
            return self._body.decode("utf-8")

    class HTTPError(Exception):
        pass

    class _ExceptionsNamespace:
        class RequestException(Exception):
            pass

        class SSLError(RequestException):
            pass

    exceptions = _ExceptionsNamespace()

    class _HTTPAdapterBase:
        def cert_verify(self, conn, url, verify, cert):
            return None

    class _AdaptersNamespace:
        HTTPAdapter = _HTTPAdapterBase

    adapters = _AdaptersNamespace()

    class Session:
        def __init__(self) -> None:
            self._mounts: Dict[str, _HTTPAdapterBase] = {}

        def mount(self, prefix: str, adapter: _HTTPAdapterBase) -> None:
            self._mounts[prefix] = adapter

        def get(self, url: str, *, headers: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> Response:
            return get(url, headers=headers, timeout=timeout)

        def post(self, url: str, data: bytes, *, headers: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> Response:
            return post(url, data, headers=headers, timeout=timeout)

    def get(url: str, *, headers: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> Response:
        req = urllib.request.Request(url, headers=headers or {}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # type: ignore[arg-type]
                body = resp.read()
                status = resp.getcode() or 0
        except Exception as exc:  # pragma: no cover - network errors
            raise HTTPError(str(exc)) from exc
        return Response(status_code=status, _body=body)

    def post(url: str, data: bytes, *, headers: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> Response:
        req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST")
        try:
            if not url.startswith(("https://", "http://")):
                raise ValueError("URL must start with 'https://' or 'http://'")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # type: ignore[arg-type]
                body = resp.read()
                status = resp.getcode() or 0
        except Exception as exc:  # pragma: no cover
            raise HTTPError(str(exc)) from exc
        return Response(status_code=status, _body=body)
