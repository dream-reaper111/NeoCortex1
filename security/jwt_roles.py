"""Role-based access control helpers for JWT protected endpoints."""

from __future__ import annotations

from typing import Iterable, Sequence

from fastapi import HTTPException, status


def has_role(user_roles: Sequence[str], required: Iterable[str]) -> bool:
    required_set = {role.lower() for role in required}
    return any(role.lower() in required_set for role in user_roles)


def enforce_roles(user_roles: Sequence[str], required: Iterable[str]) -> None:
    if not has_role(user_roles, required):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient role permissions",
        )


__all__ = ["has_role", "enforce_roles"]
