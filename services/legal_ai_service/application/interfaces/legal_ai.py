from __future__ import annotations

from typing import Protocol

from services.legal_ai_service.domain.entities import (
    AnalysisResult,
    ModelSummary,
    OpenAiCommentary,
    PredictionSnapshot,
)


class LegalAiModelGateway(Protocol):
    def analyze_text(self, text: str) -> PredictionSnapshot: ...

    def get_model_summary(self) -> ModelSummary: ...


class LegalAiCommentaryGateway(Protocol):
    def build_commentary(self, analysis_result: AnalysisResult) -> OpenAiCommentary | None: ...

