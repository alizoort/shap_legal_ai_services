from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from shap_service_runtime.config.common import (
    CommonSettings,
    LazySettingsProxy,
    get_common_settings,
    read_bool_env,
)


@dataclass(frozen=True)
class LegalAiServiceSettings(CommonSettings):
    legal_ai_database_url: str = ""
    legal_ai_model_artifacts_dir: str = ""
    legal_ai_dataset_dir: str = ""
    legal_ai_enable_commentary: bool = False
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""

    @classmethod
    def from_env(cls) -> "LegalAiServiceSettings":
        service_root = Path(__file__).resolve().parent
        return cls(
            **asdict(get_common_settings()),
            legal_ai_database_url=os.getenv(
                "LEGAL_AI_DATABASE_URL",
                os.getenv("DATABASE_URL", "postgresql://app:app@127.0.0.1:5432/legal_ai_db"),
            ),
            legal_ai_model_artifacts_dir=os.getenv(
                "LEGAL_AI_MODEL_ARTIFACTS_DIR",
                str(service_root / "model_artifacts"),
            ),
            legal_ai_dataset_dir=os.getenv(
                "LEGAL_AI_DATASET_DIR",
                str(service_root / "data"),
            ),
            legal_ai_enable_commentary=read_bool_env("LEGAL_AI_ENABLE_COMMENTARY", False),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
        )


@lru_cache(maxsize=1)
def get_settings() -> LegalAiServiceSettings:
    return LegalAiServiceSettings.from_env()


settings = LazySettingsProxy(get_settings)
