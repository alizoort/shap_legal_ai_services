from __future__ import annotations

from abc import ABC, abstractmethod

from .models import AuthResult


class AuthAdapter(ABC):
    @abstractmethod
    def authenticate(self, authorization_header: str | None) -> AuthResult:
        raise NotImplementedError
