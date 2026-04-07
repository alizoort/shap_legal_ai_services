from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignedUserContext:
    user_id: str
    roles: list[str]
    permissions: list[str]
    email: str | None = None
    claims: dict[str, Any] | None = None


@dataclass(frozen=True)
class AuthResult:
    is_authenticated: bool
    context: SignedUserContext | None = None
    raw_token: str | None = None
