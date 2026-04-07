from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from ..config.common import get_common_settings
from .context_signer import UserContextSignatureError, UserContextSigner

_SIGNED_USER_CONTEXT_HEADERS = (
    "X-User-Id",
    "X-User-Roles",
    "X-User-Permissions",
    "X-User-Email",
    "X-User-Issued-At",
    "X-User-Context-Signature",
)
_REQUIRED_SIGNED_USER_CONTEXT_HEADERS = (
    "X-User-Id",
    "X-User-Issued-At",
    "X-User-Context-Signature",
)
_LOCAL_ENVIRONMENTS = {"local", "dev", "development", "test"}


APP_ACCESS_PERMISSION = "APP_ACCESS"
LEGAL_AI_READ_PERMISSION = "LEGAL_AI_READ"
LEGAL_AI_WRITE_PERMISSION = "LEGAL_AI_WRITE"
LEGAL_AI_ADMIN_PERMISSION = "LEGAL_AI_ADMIN"


@dataclass(frozen=True)
class NormalizedUserContext:
    user_id: str
    username: str | None
    email: str | None
    permissions: list[str]
    roles: list[str]
    claims: dict[str, Any]
    source: str


def _is_local_environment() -> bool:
    settings = get_common_settings()
    return settings.environment.strip().lower() in _LOCAL_ENVIRONMENTS


def validate_signed_user_context_runtime(
    *,
    service_name: str,
    require_strict_mode: bool,
) -> None:
    settings = get_common_settings()
    secret = settings.gateway_user_context_secret.strip()
    if not secret:
        raise RuntimeError(f"{service_name}: GATEWAY_USER_CONTEXT_SECRET must be configured")

    min_secret_length = max(1, settings.gateway_user_context_secret_min_length)
    if len(secret) < min_secret_length:
        if not (
            _is_local_environment()
            and settings.allow_weak_gateway_user_context_secret_in_local
        ):
            raise RuntimeError(
                f"{service_name}: GATEWAY_USER_CONTEXT_SECRET must be at least "
                f"{min_secret_length} characters"
            )

    weak_defaults = {"dev-user-context-secret", "changeme", "change-me", "secret"}
    if secret.lower() in weak_defaults and not _is_local_environment():
        raise RuntimeError(
            f"{service_name}: insecure default GATEWAY_USER_CONTEXT_SECRET is not allowed"
        )

    if not settings.gateway_trust_user_context_headers:
        raise RuntimeError(f"{service_name}: GATEWAY_TRUST_USER_CONTEXT_HEADERS must be true")

    if require_strict_mode and not settings.require_signed_user_context:
        if not _is_local_environment():
            raise RuntimeError(f"{service_name}: REQUIRE_SIGNED_USER_CONTEXT must be true")


def _build_from_gateway_headers(
    request: Request,
    *,
    allow_missing: bool,
) -> NormalizedUserContext | None:
    settings = get_common_settings()
    if not settings.gateway_trust_user_context_headers:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Signed user context headers are not trusted",
        )

    present_headers = {
        name: request.headers.get(name, "")
        for name in _SIGNED_USER_CONTEXT_HEADERS
        if name in request.headers
    }

    if not present_headers:
        if allow_missing:
            return None
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Signed user context is required",
        )

    if len(present_headers) != len(_SIGNED_USER_CONTEXT_HEADERS):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incomplete signed user context headers",
        )

    missing_required_values = [
        name
        for name in _REQUIRED_SIGNED_USER_CONTEXT_HEADERS
        if not request.headers.get(name, "").strip()
    ]
    if missing_required_values:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incomplete signed user context headers",
        )

    signer = UserContextSigner(
        secret=settings.gateway_user_context_secret,
        max_age_seconds=settings.gateway_user_context_max_age_seconds,
    )

    try:
        verified = signer.verify(
            user_id=request.headers.get("X-User-Id", "").strip(),
            roles=request.headers.get("X-User-Roles", "").strip(),
            permissions=request.headers.get("X-User-Permissions", "").strip(),
            email=request.headers.get("X-User-Email", "").strip(),
            issued_at=request.headers.get("X-User-Issued-At", "").strip(),
            signature=request.headers.get("X-User-Context-Signature", "").strip(),
        )
    except UserContextSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid user context: {exc}",
        ) from exc

    return NormalizedUserContext(
        user_id=verified.user_id,
        username=None,
        email=verified.email,
        permissions=[item.upper() for item in verified.permissions],
        roles=[item.upper() for item in verified.roles],
        claims={},
        source="signed-headers",
    )


def get_optional_signed_user_context(request: Request) -> NormalizedUserContext | None:
    return _build_from_gateway_headers(request, allow_missing=True)


def get_required_signed_user_context(request: Request) -> NormalizedUserContext:
    context = _build_from_gateway_headers(request, allow_missing=False)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Signed user context is required",
        )
    return context


def get_normalized_user_context(request: Request) -> NormalizedUserContext:
    return get_required_signed_user_context(request)


def normalize_permissions(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    return {
        item.strip().upper()
        for item in values
        if isinstance(item, str) and item.strip()
    }


def has_all_permissions(
    current_permissions: Iterable[str] | None,
    required_permissions: Iterable[str] | None,
) -> bool:
    required = normalize_permissions(required_permissions)
    if not required:
        return True

    current = normalize_permissions(current_permissions)
    return required.issubset(current)


def has_any_permission(
    current_permissions: Iterable[str] | None,
    required_permissions: Iterable[str] | None,
) -> bool:
    required = normalize_permissions(required_permissions)
    if not required:
        return True

    current = normalize_permissions(current_permissions)
    return bool(required.intersection(current))


def user_has_all_permissions(
    user: NormalizedUserContext,
    required_permissions: Iterable[str] | None,
) -> bool:
    return has_all_permissions(user.permissions, required_permissions)


def user_has_any_permission(
    user: NormalizedUserContext,
    required_permissions: Iterable[str] | None,
) -> bool:
    return has_any_permission(user.permissions, required_permissions)


class PermissionChecker:
    def __init__(
        self,
        *required_permissions: str,
        require_all: bool = True,
    ) -> None:
        self.required_permissions = tuple(required_permissions)
        self.require_all = require_all

    def __call__(
        self,
        user: NormalizedUserContext = Depends(get_required_signed_user_context),
    ) -> NormalizedUserContext:
        if self.require_all:
            authorized = user_has_all_permissions(user, self.required_permissions)
        else:
            authorized = user_has_any_permission(user, self.required_permissions)
        if not authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user


def require_any_permission(*permissions: str) -> PermissionChecker:
    """Create a dependency that requires any of the specified permissions."""
    return PermissionChecker(*permissions, require_all=False)


def require_all_permissions(*permissions: str) -> PermissionChecker:
    """Create a dependency that requires all of the specified permissions."""
    return PermissionChecker(*permissions, require_all=True)
