from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shap_service_runtime.auth.user_context import validate_signed_user_context_runtime
from shap_service_runtime.web.observability import (
    build_service_schema_readiness_check,
    install_http_runtime_observability,
)
from services.legal_ai_service.app.controllers.legal_ai import router as legal_ai_router
from services.legal_ai_service.config import settings
from services.legal_ai_service.infrastructure.db.database import engine
from services.legal_ai_service.run_migrations import get_expected_heads


def create_app() -> FastAPI:
    validate_signed_user_context_runtime(
        service_name="legal_ai_service",
        require_strict_mode=False,
    )
    app = FastAPI(title="SHAP Legal AI Service", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_origin_regex=settings.get_cors_origin_regex(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_http_runtime_observability(
        app,
        service_name="legal_ai_service",
        readiness_check=build_service_schema_readiness_check(
            engine=engine,
            service_name="legal_ai_service",
            expected_heads=get_expected_heads(),
        ),
    )
    app.include_router(legal_ai_router)
    return app


app = create_app()
