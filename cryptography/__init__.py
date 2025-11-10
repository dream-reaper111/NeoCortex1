"""Minimal subset of the :mod:`cryptography` package used in tests."""
from __future__ import annotations

from .fernet import Fernet, InvalidToken  # noqa: F401

__all__ = ["Fernet", "InvalidToken"]
