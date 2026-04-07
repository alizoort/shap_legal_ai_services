from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from services.legal_ai_service.application.services.legal_ai_scaffold_service import (
    LegalAiScaffoldService,
)

router = APIRouter(prefix="/legal-ai", tags=["legal-ai"])


class LegalAiPingResponseModel(BaseModel):
    service_name: str
    status: str
    message: str
    reserved_routes: list[str]


@router.get("/ping", response_model=LegalAiPingResponseModel)
def ping() -> LegalAiPingResponseModel:
    response = LegalAiScaffoldService().build_ping_response()
    return LegalAiPingResponseModel(
        service_name=response.service_name,
        status=response.status,
        message=response.message,
        reserved_routes=list(response.reserved_routes),
    )
