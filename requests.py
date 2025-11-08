from __future__ import annotations

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
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # type: ignore[arg-type]
            body = resp.read()
            status = resp.getcode() or 0
    except Exception as exc:  # pragma: no cover
        raise HTTPError(str(exc)) from exc
    return Response(status_code=status, _body=body)

