"""JWT helpers and authentication dependencies for the NeoCortex platform."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer

HTTP_401_UNAUTHORIZED = 401

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login", auto_error=False)


def _server_module():
    import server  # type: ignore

    return server


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Encode a JWT access token using the server's configured settings."""

    server = _server_module()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=server.JWT_ACCESS_EXPIRE_MINUTES))
    payload = data.copy()
    payload.setdefault("iat", int(now.timestamp()))
    payload.setdefault("nbf", int(now.timestamp()))
    payload.setdefault("exp", int(expire.timestamp()))
    return jwt.encode(payload, server.JWT_SECRET_KEY, algorithm=server.JWT_ALGORITHM)


async def get_current_user(
    request: Request, token: Optional[str] = Depends(_oauth2_scheme)
) -> Dict[str, Any]:
    """Resolve the authenticated user from the current request context."""

    server = _server_module()
    payload = getattr(request.state, "access_token_payload", None)
    raw_token: Optional[str] = token

    if raw_token is None:
        authorization = request.headers.get("Authorization")
        if authorization:
            scheme, _, credentials = authorization.partition(" ")
            if scheme.lower() == "bearer" and credentials:
                raw_token = credentials.strip()
    if raw_token is None:
        raw_token = request.cookies.get(server.SESSION_COOKIE_NAME)

    if payload is None:
        if not raw_token:
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        payload = server._decode_access_token(raw_token)
    elif raw_token is None:
        raw_token = request.cookies.get(server.SESSION_COOKIE_NAME)

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid token subject")

    try:
        user = server._load_user_record(int(user_id))
    except Exception as exc:  # pragma: no cover - delegated to server helper
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Unknown user") from exc

    request.state.current_user = user
    request.state.access_token_payload = payload

    result: Dict[str, Any] = {
        "id": user.get("id"),
        "username": user.get("username"),
        "roles": user.get("roles", []),
        "scopes": user.get("scopes", []),
    }
    if raw_token:
        result["token"] = raw_token
    result["token_payload"] = payload
    return result


__all__ = ["create_access_token", "get_current_user"]
