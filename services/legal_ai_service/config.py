from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache

from shap_service_runtime.config.common import CommonSettings, LazySettingsProxy, get_common_settings


@dataclass(frozen=True)
class LegalAiServiceSettings(CommonSettings):
    legal_ai_database_url: str = ""

    @classmethod
    def from_env(cls) -> "LegalAiServiceSettings":
        return cls(
            **asdict(get_common_settings()),
            legal_ai_database_url=os.getenv(
                "LEGAL_AI_DATABASE_URL",
                os.getenv("DATABASE_URL", "postgresql://app:app@127.0.0.1:5432/legal_ai_db"),
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> LegalAiServiceSettings:
    return LegalAiServiceSettings.from_env()


settings = LazySettingsProxy(get_settings)
