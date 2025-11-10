"""Minimal PyJWT-compatible subset for offline test environments."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

__all__ = [
    "encode",
    "decode",
    "InvalidTokenError",
    "ExpiredSignatureError",
]


class InvalidTokenError(Exception):
    """Raised when a token cannot be decoded or validated."""


class ExpiredSignatureError(InvalidTokenError):
    """Raised when a token's ``exp`` claim is in the past."""


_HASH_ALGORITHMS = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def encode(
    payload: Mapping[str, Any],
    key: str,
    *,
    algorithm: str = "HS256",
    headers: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return a signed JWT string using an HMAC-based algorithm."""

    if algorithm not in _HASH_ALGORITHMS:
        raise InvalidTokenError(f"Unsupported algorithm: {algorithm}")

    header: Dict[str, Any] = {"typ": "JWT", "alg": algorithm}
    if headers:
        header.update(dict(headers))

    header_segment = _b64encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_segment = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")

    mac = hmac.new(key.encode("utf-8"), signing_input, _HASH_ALGORITHMS[algorithm])
    signature_segment = _b64encode(mac.digest())
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def decode(
    token: str,
    key: str,
    *,
    algorithms: Optional[Iterable[str]] = None,
    audience: Optional[str] = None,
    issuer: Optional[str] = None,
    leeway: int | float = 0,
    options: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate and decode a JWT string.

    Only behaviour used inside ``server.py`` is implemented.  Unsupported
    options raise :class:`InvalidTokenError` to surface misconfiguration in
    tests.
    """

    if not token:
        raise InvalidTokenError("Token is empty")

    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidTokenError("Not enough segments")

    header_bytes = _b64decode(parts[0])
    payload_bytes = _b64decode(parts[1])
    signature = parts[2]

    try:
        header = json.loads(header_bytes.decode("utf-8"))
        payload: Dict[str, Any] = json.loads(payload_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise InvalidTokenError("Invalid JSON payload") from exc

    algorithm = header.get("alg")
    if algorithms is not None and algorithm not in set(algorithms):
        raise InvalidTokenError("Algorithm not allowed")
    if algorithm not in _HASH_ALGORITHMS:
        raise InvalidTokenError(f"Unsupported algorithm: {algorithm}")

    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected_sig = _b64encode(hmac.new(key.encode("utf-8"), signing_input, _HASH_ALGORITHMS[algorithm]).digest())
    if not hmac.compare_digest(signature, expected_sig):
        raise InvalidTokenError("Signature verification failed")

    now = time.time()
    verify_exp = True
    verify_nbf = True
    verify_iat = True
    if options:
        verify_exp = options.get("verify_exp", True)
        verify_nbf = options.get("verify_nbf", True)
        verify_iat = options.get("verify_iat", True)
        unsupported = set(options).difference({"verify_exp", "verify_signature", "verify_nbf", "verify_iat"})
        if unsupported:
            raise InvalidTokenError(f"Unsupported options: {', '.join(sorted(unsupported))}")

    if verify_exp and "exp" in payload:
        if float(payload["exp"]) + float(leeway) < now:
            raise ExpiredSignatureError("Token has expired")

    if verify_nbf and "nbf" in payload:
        if float(payload["nbf"]) - float(leeway) > now:
            raise InvalidTokenError("Token not yet valid")

    if verify_iat and "iat" in payload:
        if float(payload["iat"]) - float(leeway) > now:
            raise InvalidTokenError("Token used before issued")

    if audience is not None and payload.get("aud") != audience:
        raise InvalidTokenError("Invalid audience")

    if issuer is not None and payload.get("iss") != issuer:
        raise InvalidTokenError("Invalid issuer")

    return payload
