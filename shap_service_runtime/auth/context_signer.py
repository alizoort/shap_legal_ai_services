from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import SignedUserContext


class UserContextSignatureError(Exception):
    pass


@dataclass(frozen=True)
class SignedHeaders:
    user_id: str
    roles: str
    permissions: str
    email: str
    issued_at: str
    signature: str


class UserContextSigner:
    def __init__(self, secret: str, max_age_seconds: int = 60) -> None:
        if not secret:
            raise ValueError("Signer secret cannot be empty")
        self.secret = secret.encode("utf-8")
        self.max_age_seconds = max_age_seconds

    def _canonical(
        self,
        *,
        user_id: str,
        roles: str,
        permissions: str,
        email: str,
        issued_at: str,
    ) -> str:
        return (
            f"user_id={user_id}\n"
            f"roles={roles}\n"
            f"permissions={permissions}\n"
            f"email={email}\n"
            f"issued_at={issued_at}"
        )

    def _sign(self, canonical_payload: str) -> str:
        digest = hmac.new(
            self.secret,
            canonical_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return digest

    def build_signed_headers(
        self,
        context: SignedUserContext,
        issued_at: int | None = None,
    ) -> SignedHeaders:
        issued = str(issued_at or int(datetime.now(timezone.utc).timestamp()))
        roles = ",".join(sorted({item.strip().upper() for item in context.roles if item and item.strip()}))
        permissions = ",".join(
            sorted({item.strip().upper() for item in context.permissions if item and item.strip()})
        )
        email = (context.email or "").strip()
        canonical_payload = self._canonical(
            user_id=context.user_id,
            roles=roles,
            permissions=permissions,
            email=email,
            issued_at=issued,
        )
        signature = self._sign(canonical_payload)
        return SignedHeaders(
            user_id=context.user_id,
            roles=roles,
            permissions=permissions,
            email=email,
            issued_at=issued,
            signature=signature,
        )

    def verify(
        self,
        *,
        user_id: str,
        roles: str,
        permissions: str,
        email: str,
        issued_at: str,
        signature: str,
    ) -> SignedUserContext:
        if not issued_at.isdigit():
            raise UserContextSignatureError("Invalid issued_at header")
        issued_ts = int(issued_at)
        now_ts = int(datetime.now(timezone.utc).timestamp())
        if abs(now_ts - issued_ts) > self.max_age_seconds:
            raise UserContextSignatureError("User context signature expired")
        canonical_payload = self._canonical(
            user_id=user_id,
            roles=roles,
            permissions=permissions,
            email=email,
            issued_at=issued_at,
        )
        expected = self._sign(canonical_payload)
        if not hmac.compare_digest(expected, signature):
            raise UserContextSignatureError("Invalid user context signature")
        parsed_roles = [item.strip() for item in roles.split(",") if item.strip()]
        parsed_permissions = [item.strip() for item in permissions.split(",") if item.strip()]
        return SignedUserContext(
            user_id=user_id,
            roles=parsed_roles,
            permissions=parsed_permissions,
            email=email or None,
            claims=None,
        )
