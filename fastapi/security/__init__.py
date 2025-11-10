"""Minimal security primitives for the FastAPI stub used in tests."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

from .. import HTTPException, Request


@dataclass
class HTTPBasicCredentials:
    """Container for HTTP Basic authentication credentials."""

    username: str
    password: str


class HTTPBasic:
    """Parse HTTP Basic credentials from the Authorization header."""

    def __init__(self, *, auto_error: bool = True) -> None:
        self.auto_error = auto_error

    async def __call__(self, request: Request) -> Optional[HTTPBasicCredentials]:
        header = request.headers.get("authorization")
        if not header or not header.lower().startswith("basic "):
            if self.auto_error:
                raise HTTPException(
                    401,
                    "Not authenticated",
                    headers={"WWW-Authenticate": "Basic"},
                )
            return None

        encoded = header.split(" ", 1)[1]
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except Exception:
            if self.auto_error:
                raise HTTPException(
                    400,
                    "Invalid basic auth header",
                    headers={"WWW-Authenticate": "Basic"},
                )
            return None

        if ":" not in decoded:
            if self.auto_error:
                raise HTTPException(
                    400,
                    "Invalid basic auth header",
                    headers={"WWW-Authenticate": "Basic"},
                )
            return None

        username, password = decoded.split(":", 1)
        return HTTPBasicCredentials(username=username, password=password)


class OAuth2PasswordBearer:
    """Minimal token extractor used for bearer authentication in tests."""

    def __init__(self, tokenUrl: str, *, auto_error: bool = True) -> None:
        self.tokenUrl = tokenUrl
        self.auto_error = auto_error

    async def __call__(self, request: Request) -> Optional[str]:
        header = request.headers.get("authorization")
        if header and header.lower().startswith("bearer "):
            return header.split(" ", 1)[1]
        if self.auto_error:
            raise HTTPException(
                401,
                "Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None


__all__ = ["HTTPBasic", "HTTPBasicCredentials", "OAuth2PasswordBearer"]
