from __future__ import annotations

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware

from shap_service_runtime.auth.user_context import validate_signed_user_context_runtime
from shap_service_runtime.web.observability import install_http_runtime_observability
from services.gateway_service.app.proxy import build_proxy_client, proxy_request
from services.gateway_service.app.security import build_signed_user_headers
from services.gateway_service.config import settings

_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]


def create_app() -> FastAPI:
    validate_signed_user_context_runtime(
        service_name="gateway_service",
        require_strict_mode=False,
    )
    app = FastAPI(title="SHAP API Gateway", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_origin_regex=settings.get_cors_origin_regex(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def readiness_check() -> dict[str, str]:
        client: httpx.AsyncClient | None = getattr(app.state, "proxy_client", None)
        if client is None:
            raise RuntimeError("Gateway proxy client is not initialized.")
        return {"dependency": "proxy_client", "status": "ok"}

    install_http_runtime_observability(
        app,
        service_name="gateway_service",
        readiness_check=readiness_check,
    )

    @app.get("/gateway/context")
    def gateway_context(authorization: str | None = Header(default=None)) -> dict[str, str]:
        return build_signed_user_headers(authorization)

    @app.on_event("startup")
    async def startup_proxy_client() -> None:
        app.state.proxy_client = build_proxy_client()

    @app.on_event("shutdown")
    async def shutdown_proxy_client() -> None:
        client: httpx.AsyncClient | None = getattr(app.state, "proxy_client", None)
        if client is not None:
            await client.aclose()

    async def _proxy_to_legal_ai(request: Request):
        client: httpx.AsyncClient | None = getattr(request.app.state, "proxy_client", None)
        if client is None:
            raise RuntimeError("Gateway proxy client is not initialized.")
        return await proxy_request(
            client=client,
            request=request,
            upstream_base_url=settings.legal_ai_service_base_url,
            extra_headers=build_signed_user_headers(request.headers.get("authorization")),
        )

    @app.api_route("/legal-ai", methods=_PROXY_METHODS)
    @app.api_route("/legal-ai/{proxy_path:path}", methods=_PROXY_METHODS)
    async def proxy_legal_ai(request: Request, proxy_path: str = ""):
        _ = proxy_path
        return await _proxy_to_legal_ai(request)

    return app


app = create_app()
