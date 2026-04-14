from __future__ import annotations

from dataclasses import replace

from services.legal_ai_service.application.dtos.legal_ai import (
    LegalAiAnalyzeRequest,
    LegalAiPingResponse,
)
from services.legal_ai_service.application.interfaces import (
    LegalAiCommentaryGateway,
    LegalAiModelGateway,
)
from services.legal_ai_service.domain.entities import AnalysisResult, ModelSummary


class LegalAiAnalysisService:
    def __init__(
        self,
        *,
        model_gateway: LegalAiModelGateway,
        commentary_gateway: LegalAiCommentaryGateway,
    ) -> None:
        self._model_gateway = model_gateway
        self._commentary_gateway = commentary_gateway

    def build_ping_response(self) -> LegalAiPingResponse:
        return LegalAiPingResponse(
            service_name="legal_ai_service",
            status="ok",
            message="Employment compliance risk analysis routes are reachable.",
            reserved_routes=(
                "/legal-ai/ping",
                "/legal-ai/model-summary",
                "/legal-ai/analyze",
            ),
        )

    def analyze(self, request: LegalAiAnalyzeRequest) -> AnalysisResult:
        prediction = self._model_gateway.analyze_text(request.text.strip())
        result = AnalysisResult(
            risk_label=prediction.risk_label,
            class_probabilities=prediction.class_probabilities,
            needs_human_review=(
                prediction.risk_label != "low"
                or prediction.class_probabilities.max_probability() < 0.60
            ),
            top_features_supporting_prediction=prediction.top_features_supporting_prediction,
            top_features_against_prediction=prediction.top_features_against_prediction,
            plain_english_explanation=prediction.plain_english_explanation,
            openai_commentary=None,
        )
        if not request.include_commentary:
            return result
        return replace(
            result,
            openai_commentary=self._commentary_gateway.build_commentary(result),
        )

    def get_model_summary(self) -> ModelSummary:
        return self._model_gateway.get_model_summary()

