from __future__ import annotations

from services.legal_ai_service.application.dtos.legal_ai import LegalAiPingResponse


class LegalAiScaffoldService:
    def build_ping_response(self) -> LegalAiPingResponse:
        return LegalAiPingResponse(
            service_name="legal_ai_service",
            status="ok",
            message="Reserved SHAP Legal AI route is reachable.",
            reserved_routes=("/legal-ai/ping", "/legal-ai/*"),
        )
