from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional, TypeVar

_T = TypeVar("_T")


def read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def read_optional_float_env(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_positive_int_csv_env(name: str, default_csv: str) -> tuple[int, ...]:
    raw = os.getenv(name, default_csv)
    if raw is None:
        raise ValueError(f"{name} must be configured with comma-separated positive integers")

    values: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(f"{name} contains non-integer value: {token}") from exc
        if parsed <= 0:
            raise ValueError(f"{name} contains non-positive value: {parsed}")
        values.append(parsed)

    if not values:
        raise ValueError(f"{name} must contain at least one positive integer delay")

    return tuple(values)


class LazySettingsProxy:
    def __init__(self, loader: Callable[[], _T]) -> None:
        self._loader = loader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader(), name)

    def __repr__(self) -> str:
        return repr(self._loader())


@dataclass(frozen=True)
class MessagingSettings:
    """RabbitMQ messaging topology and retry policy configuration."""

    url: str
    exchange: str
    legal_ai_events_queue: str
    publish_batch_size: int
    prefetch_count: int
    connect_timeout_seconds: float
    heartbeat_seconds: int
    poll_interval_seconds: float
    retry_delays_ms: tuple[int, ...]
    inbox_max_retries: int
    dlq_suffix: str
    retry_suffix: str

    @classmethod
    def from_env(cls) -> "MessagingSettings":
        return cls(
            url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672//"),
            exchange=os.getenv("RABBITMQ_EXCHANGE", "shap.events"),
            legal_ai_events_queue=os.getenv(
                "RABBITMQ_LEGAL_AI_EVENTS_QUEUE",
                "shap.legal_ai.events",
            ),
            publish_batch_size=read_int_env("RABBITMQ_PUBLISH_BATCH_SIZE", 100),
            prefetch_count=read_int_env("RABBITMQ_PREFETCH_COUNT", 20),
            connect_timeout_seconds=read_float_env("RABBITMQ_CONNECT_TIMEOUT_SECONDS", 5.0),
            heartbeat_seconds=read_int_env("RABBITMQ_HEARTBEAT_SECONDS", 30),
            poll_interval_seconds=read_float_env("RABBITMQ_POLL_INTERVAL_SECONDS", 1.0),
            retry_delays_ms=read_positive_int_csv_env(
                "RABBITMQ_RETRY_DELAYS_MS",
                "5000,30000,120000",
            ),
            inbox_max_retries=read_int_env("RABBITMQ_INBOX_MAX_RETRIES", 3),
            dlq_suffix=os.getenv("RABBITMQ_DLQ_SUFFIX", ".dlq"),
            retry_suffix=os.getenv("RABBITMQ_RETRY_SUFFIX", ".retry"),
        )

    def __post_init__(self) -> None:
        if not self.retry_delays_ms:
            raise ValueError("RABBITMQ_RETRY_DELAYS_MS must contain at least one delay value")
        if any(delay <= 0 for delay in self.retry_delays_ms):
            raise ValueError("RABBITMQ_RETRY_DELAYS_MS must contain only positive delays")
        if self.inbox_max_retries <= 0:
            raise ValueError("RABBITMQ_INBOX_MAX_RETRIES must be a positive integer")
        if self.inbox_max_retries != len(self.retry_delays_ms):
            raise ValueError(
                "RABBITMQ_INBOX_MAX_RETRIES must equal the number of "
                "RABBITMQ_RETRY_DELAYS_MS stages"
            )
        if not self.dlq_suffix.strip():
            raise ValueError("RABBITMQ_DLQ_SUFFIX must not be empty")
        if not self.retry_suffix.strip():
            raise ValueError("RABBITMQ_RETRY_SUFFIX must not be empty")


@dataclass(frozen=True)
class AuthSettings:
    """Gateway user context signing and validation configuration."""

    secret: str
    secret_min_length: int
    allow_weak_secret_in_local: bool
    max_age_seconds: int
    trust_headers: bool
    require_signed_context: bool

    @classmethod
    def from_env(cls) -> "AuthSettings":
        return cls(
            secret=os.getenv(
                "GATEWAY_USER_CONTEXT_SECRET",
                "shap-legal-ai-local-user-context-secret",
            ),
            secret_min_length=read_int_env("GATEWAY_USER_CONTEXT_SECRET_MIN_LENGTH", 24),
            allow_weak_secret_in_local=read_bool_env(
                "ALLOW_WEAK_GATEWAY_USER_CONTEXT_SECRET_IN_LOCAL",
                True,
            ),
            max_age_seconds=read_int_env("GATEWAY_USER_CONTEXT_MAX_AGE_SECONDS", 60),
            trust_headers=read_bool_env("GATEWAY_TRUST_USER_CONTEXT_HEADERS", True),
            require_signed_context=read_bool_env("REQUIRE_SIGNED_USER_CONTEXT", True),
        )


@dataclass(frozen=True)
class ObservabilitySettings:
    """CORS and observability configuration."""

    cors_origins: str
    cors_allow_origin_regex: str
    environment: str

    @classmethod
    def from_env(cls, environment: str | None = None) -> "ObservabilitySettings":
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "local")
        return cls(
            cors_origins=os.getenv(
                "CORS_ORIGINS",
                (
                    "http://localhost:4200,http://127.0.0.1:4200,"
                    "http://localhost:3000,http://127.0.0.1:3000"
                ),
            ),
            cors_allow_origin_regex=os.getenv("CORS_ALLOW_ORIGIN_REGEX", ""),
            environment=environment,
        )

    def get_cors_origins(self) -> list[str]:
        if not self.cors_origins:
            return []
        origins: list[str] = []
        for origin in self.cors_origins.split(","):
            normalized = origin.strip()
            if not normalized:
                continue
            if normalized != "*":
                normalized = normalized.rstrip("/")
            origins.append(normalized)
        return origins

    def get_cors_origin_regex(self) -> str | None:
        if self.cors_allow_origin_regex:
            return self.cors_allow_origin_regex
        if self.environment.lower() in {"local", "dev", "development"}:
            return r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
        return None


@dataclass(frozen=True)
class CommonSettings:
    """
    Legacy unified settings class for backward compatibility.

    DEPRECATED: Prefer using domain-specific settings classes:
    - MessagingSettings for RabbitMQ configuration
    - AuthSettings for user context signing
    - ObservabilitySettings for CORS configuration
    """

    app_name: str
    environment: str
    rabbitmq_url: str
    rabbitmq_exchange: str
    rabbitmq_legal_ai_events_queue: str
    rabbitmq_publish_batch_size: int
    rabbitmq_prefetch_count: int
    rabbitmq_connect_timeout_seconds: float
    rabbitmq_heartbeat_seconds: int
    rabbitmq_poll_interval_seconds: float
    rabbitmq_retry_delays_ms: tuple[int, ...]
    rabbitmq_inbox_max_retries: int
    rabbitmq_dlq_suffix: str
    rabbitmq_retry_suffix: str
    cors_origins: str
    cors_allow_origin_regex: str
    gateway_user_context_secret: str
    gateway_user_context_secret_min_length: int
    allow_weak_gateway_user_context_secret_in_local: bool
    gateway_user_context_max_age_seconds: int
    gateway_trust_user_context_headers: bool
    require_signed_user_context: bool

    @classmethod
    def from_env(cls) -> "CommonSettings":
        return cls(
            app_name=os.getenv("APP_NAME", "SHAP Legal AI"),
            environment=os.getenv("ENVIRONMENT", "local"),
            rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672//"),
            rabbitmq_exchange=os.getenv("RABBITMQ_EXCHANGE", "shap.events"),
            rabbitmq_legal_ai_events_queue=os.getenv(
                "RABBITMQ_LEGAL_AI_EVENTS_QUEUE",
                "shap.legal_ai.events",
            ),
            rabbitmq_publish_batch_size=read_int_env("RABBITMQ_PUBLISH_BATCH_SIZE", 100),
            rabbitmq_prefetch_count=read_int_env("RABBITMQ_PREFETCH_COUNT", 20),
            rabbitmq_connect_timeout_seconds=read_float_env(
                "RABBITMQ_CONNECT_TIMEOUT_SECONDS",
                5.0,
            ),
            rabbitmq_heartbeat_seconds=read_int_env("RABBITMQ_HEARTBEAT_SECONDS", 30),
            rabbitmq_poll_interval_seconds=read_float_env(
                "RABBITMQ_POLL_INTERVAL_SECONDS",
                1.0,
            ),
            rabbitmq_retry_delays_ms=read_positive_int_csv_env(
                "RABBITMQ_RETRY_DELAYS_MS",
                "5000,30000,120000",
            ),
            rabbitmq_inbox_max_retries=read_int_env("RABBITMQ_INBOX_MAX_RETRIES", 3),
            rabbitmq_dlq_suffix=os.getenv("RABBITMQ_DLQ_SUFFIX", ".dlq"),
            rabbitmq_retry_suffix=os.getenv("RABBITMQ_RETRY_SUFFIX", ".retry"),
            cors_origins=os.getenv(
                "CORS_ORIGINS",
                (
                    "http://localhost:4200,http://127.0.0.1:4200,"
                    "http://localhost:3000,http://127.0.0.1:3000"
                ),
            ),
            cors_allow_origin_regex=os.getenv("CORS_ALLOW_ORIGIN_REGEX", ""),
            gateway_user_context_secret=os.getenv(
                "GATEWAY_USER_CONTEXT_SECRET",
                "shap-legal-ai-local-user-context-secret",
            ),
            gateway_user_context_secret_min_length=read_int_env(
                "GATEWAY_USER_CONTEXT_SECRET_MIN_LENGTH",
                24,
            ),
            allow_weak_gateway_user_context_secret_in_local=read_bool_env(
                "ALLOW_WEAK_GATEWAY_USER_CONTEXT_SECRET_IN_LOCAL",
                True,
            ),
            gateway_user_context_max_age_seconds=read_int_env(
                "GATEWAY_USER_CONTEXT_MAX_AGE_SECONDS",
                60,
            ),
            gateway_trust_user_context_headers=read_bool_env(
                "GATEWAY_TRUST_USER_CONTEXT_HEADERS",
                True,
            ),
            require_signed_user_context=read_bool_env(
                "REQUIRE_SIGNED_USER_CONTEXT",
                True,
            ),
        )

    def __post_init__(self) -> None:
        if not self.rabbitmq_retry_delays_ms:
            raise ValueError("RABBITMQ_RETRY_DELAYS_MS must contain at least one delay value")
        if any(delay <= 0 for delay in self.rabbitmq_retry_delays_ms):
            raise ValueError("RABBITMQ_RETRY_DELAYS_MS must contain only positive delays")
        if self.rabbitmq_inbox_max_retries <= 0:
            raise ValueError("RABBITMQ_INBOX_MAX_RETRIES must be a positive integer")
        if self.rabbitmq_inbox_max_retries != len(self.rabbitmq_retry_delays_ms):
            raise ValueError(
                "RABBITMQ_INBOX_MAX_RETRIES must equal the number of "
                "RABBITMQ_RETRY_DELAYS_MS stages"
            )
        if not self.rabbitmq_dlq_suffix.strip():
            raise ValueError("RABBITMQ_DLQ_SUFFIX must not be empty")
        if not self.rabbitmq_retry_suffix.strip():
            raise ValueError("RABBITMQ_RETRY_SUFFIX must not be empty")

    def get_cors_origins(self) -> list[str]:
        if not self.cors_origins:
            return []
        origins: list[str] = []
        for origin in self.cors_origins.split(","):
            normalized = origin.strip()
            if not normalized:
                continue
            if normalized != "*":
                normalized = normalized.rstrip("/")
            origins.append(normalized)
        return origins

    def get_cors_origin_regex(self) -> str | None:
        if self.cors_allow_origin_regex:
            return self.cors_allow_origin_regex
        if self.environment.lower() in {"local", "dev", "development"}:
            return r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
        return None


@lru_cache(maxsize=1)
def get_common_settings() -> CommonSettings:
    """
    Get unified settings instance (legacy, for backward compatibility).

    DEPRECATED: Prefer using domain-specific getters:
    - get_messaging_settings() for RabbitMQ configuration
    - get_auth_settings() for user context signing
    - get_observability_settings() for CORS configuration
    """
    return CommonSettings.from_env()


@lru_cache(maxsize=1)
def get_messaging_settings() -> MessagingSettings:
    """Get RabbitMQ messaging topology and retry policy configuration."""
    return MessagingSettings.from_env()


@lru_cache(maxsize=1)
def get_auth_settings() -> AuthSettings:
    """Get gateway user context signing and validation configuration."""
    return AuthSettings.from_env()


@lru_cache(maxsize=1)
def get_observability_settings() -> ObservabilitySettings:
    """Get CORS and observability configuration."""
    return ObservabilitySettings.from_env()
