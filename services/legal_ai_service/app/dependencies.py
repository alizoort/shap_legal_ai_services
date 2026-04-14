from __future__ import annotations

from functools import lru_cache

from services.legal_ai_service.application.services.legal_ai_analysis_service import (
    LegalAiAnalysisService,
)
from services.legal_ai_service.infrastructure.llm.openai_commentary_client import (
    build_commentary_gateway,
)
from services.legal_ai_service.infrastructure.ml.legal_ai_model_gateway import (
    SklearnLegalAiModelGateway,
)


@lru_cache(maxsize=1)
def get_legal_ai_analysis_service() -> LegalAiAnalysisService:
    return LegalAiAnalysisService(
        model_gateway=SklearnLegalAiModelGateway(),
        commentary_gateway=build_commentary_gateway(),
    )

