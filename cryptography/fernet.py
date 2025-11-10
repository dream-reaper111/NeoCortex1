"""Tiny stand-in for :mod:`cryptography.fernet` used in offline tests."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass

__all__ = ["Fernet", "InvalidToken"]


class InvalidToken(Exception):
    """Raised when token data cannot be decrypted."""


@dataclass
class Fernet:
    _key: bytes

    def __init__(self, key: bytes | str) -> None:
        if isinstance(key, str):
            key = key.encode("utf-8")
        if not key:
            raise ValueError("Fernet key must not be empty")
        try:
            # Validate that the key decodes as base64; keep raw bytes for deterministic results.
            base64.urlsafe_b64decode(key)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid base64 key") from exc
        self._key = key

    @staticmethod
    def generate_key() -> bytes:
        random_bytes = os.urandom(32)
        return base64.urlsafe_b64encode(random_bytes)

    def encrypt(self, data: bytes) -> bytes:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        nonce = os.urandom(16)
        payload = nonce + bytes(data)
        return base64.urlsafe_b64encode(payload)

    def decrypt(self, token: bytes) -> bytes:
        if isinstance(token, str):
            token_bytes = token.encode("utf-8")
        else:
            token_bytes = bytes(token)
        try:
            decoded = base64.urlsafe_b64decode(token_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            raise InvalidToken("Token is not valid base64") from exc
        if len(decoded) < 16:
            raise InvalidToken("Token is too short")
        return decoded[16:]
