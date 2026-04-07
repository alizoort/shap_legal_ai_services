from __future__ import annotations

from shap_service_runtime.auth.context_signer import UserContextSigner
from shap_service_runtime.auth.models import SignedUserContext
from services.gateway_service.config import settings

signer = UserContextSigner(
    secret=settings.gateway_user_context_secret,
    max_age_seconds=settings.gateway_user_context_max_age_seconds,
)


def build_signed_user_headers(authorization: str | None) -> dict[str, str]:
    _ = authorization
    signed = signer.build_signed_headers(
        SignedUserContext(
            user_id="anonymous-dev",
            roles=["ANONYMOUS"],
            permissions=["APP_ACCESS"],
            email=None,
        )
    )
    return {
        "X-User-Id": signed.user_id,
        "X-User-Roles": signed.roles,
        "X-User-Permissions": signed.permissions,
        "X-User-Email": signed.email,
        "X-User-Issued-At": signed.issued_at,
        "X-User-Context-Signature": signed.signature,
    }
