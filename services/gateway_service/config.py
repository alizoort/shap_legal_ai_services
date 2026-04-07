from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache

from shap_service_runtime.config.common import (
    CommonSettings,
    LazySettingsProxy,
    get_common_settings,
    read_float_env,
    read_int_env,
)


@dataclass(frozen=True)
class GatewaySettings(CommonSettings):
    legal_ai_service_base_url: str = ""
    gateway_proxy_connect_timeout_seconds: float = 0.0
    gateway_proxy_read_timeout_seconds: float = 0.0
    gateway_proxy_write_timeout_seconds: float = 0.0
    gateway_proxy_pool_timeout_seconds: float = 0.0
    gateway_proxy_max_connections: int = 0
    gateway_proxy_max_keepalive_connections: int = 0

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            **asdict(get_common_settings()),
            legal_ai_service_base_url=os.getenv(
                "LEGAL_AI_SERVICE_BASE_URL",
                "http://legal-ai-service:8103",
            ),
            gateway_proxy_connect_timeout_seconds=read_float_env(
                "GATEWAY_PROXY_CONNECT_TIMEOUT_SECONDS",
                5.0,
            ),
            gateway_proxy_read_timeout_seconds=read_float_env(
                "GATEWAY_PROXY_READ_TIMEOUT_SECONDS",
                30.0,
            ),
            gateway_proxy_write_timeout_seconds=read_float_env(
                "GATEWAY_PROXY_WRITE_TIMEOUT_SECONDS",
                30.0,
            ),
            gateway_proxy_pool_timeout_seconds=read_float_env(
                "GATEWAY_PROXY_POOL_TIMEOUT_SECONDS",
                5.0,
            ),
            gateway_proxy_max_connections=read_int_env(
                "GATEWAY_PROXY_MAX_CONNECTIONS",
                100,
            ),
            gateway_proxy_max_keepalive_connections=read_int_env(
                "GATEWAY_PROXY_MAX_KEEPALIVE_CONNECTIONS",
                20,
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> GatewaySettings:
    return GatewaySettings.from_env()


settings = LazySettingsProxy(get_settings)
