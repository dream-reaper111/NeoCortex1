from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_third_party_requests():
    """Attempt to load the real 'requests' package, skipping this stub."""

    this_dir = Path(__file__).resolve().parent
    for entry in list(sys.path):
        try:
            if not entry:
                continue
            if Path(entry).resolve() == this_dir:
                continue
        except Exception:
            # Some entries may not be valid paths; ignore them.
            continue

        spec = importlib.util.find_spec("requests", [entry])
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


_third_party_requests = _load_third_party_requests()

if _third_party_requests is not None:
    sys.modules[__name__] = _third_party_requests
    globals().update(_third_party_requests.__dict__)
else:
    import json
    import urllib.request
    from dataclasses import dataclass
    from typing import Any, Dict, Optional


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
            if not url.startswith(('https://', 'http://')):
                raise ValueError("URL must start with 'https://' or 'http://'")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # type: ignore[arg-type]
                body = resp.read()
                status = resp.getcode() or 0
        except Exception as exc:  # pragma: no cover
            raise HTTPError(str(exc)) from exc
        return Response(status_code=status, _body=body)

