from __future__ import annotations

from collections.abc import Mapping

import httpx
from fastapi import HTTPException, Request, status
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from services.gateway_service.config import settings

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

_STRIPPED_REQUEST_HEADERS = {
    "authorization",
    "host",
    "content-length",
}

_STRIPPED_RESPONSE_HEADERS = {
    "content-length",
}

_STRIPPED_PREFIXES = (
    "x-user-",
    "x-forwarded-",
)


def build_proxy_client() -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        connect=settings.gateway_proxy_connect_timeout_seconds,
        read=settings.gateway_proxy_read_timeout_seconds,
        write=settings.gateway_proxy_write_timeout_seconds,
        pool=settings.gateway_proxy_pool_timeout_seconds,
    )
    limits = httpx.Limits(
        max_connections=settings.gateway_proxy_max_connections,
        max_keepalive_connections=settings.gateway_proxy_max_keepalive_connections,
    )
    return httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=False)


def sanitize_request_headers(
    headers: Mapping[str, str],
    *,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in _HOP_BY_HOP_HEADERS or lowered in _STRIPPED_REQUEST_HEADERS:
            continue
        if lowered.startswith(_STRIPPED_PREFIXES):
            continue
        forwarded[key] = value
    if extra_headers:
        forwarded.update(extra_headers)
    return forwarded


def sanitize_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in _HOP_BY_HOP_HEADERS or lowered in _STRIPPED_RESPONSE_HEADERS:
            continue
        if lowered.startswith(_STRIPPED_PREFIXES):
            continue
        forwarded[key] = value
    return forwarded


def build_upstream_url(*, upstream_base_url: str, path: str, query_string: str | None) -> str:
    base = upstream_base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    if query_string:
        return f"{base}{normalized_path}?{query_string}"
    return f"{base}{normalized_path}"


async def proxy_request(
    *,
    client: httpx.AsyncClient,
    request: Request,
    upstream_base_url: str,
    extra_headers: Mapping[str, str] | None = None,
) -> StreamingResponse:
    body = await request.body()
    outbound_headers = sanitize_request_headers(request.headers, extra_headers=extra_headers)
    upstream_url = build_upstream_url(
        upstream_base_url=upstream_base_url,
        path=request.url.path,
        query_string=request.url.query,
    )
    outbound_request = client.build_request(
        request.method,
        upstream_url,
        headers=outbound_headers,
        content=body or None,
    )
    try:
        upstream_response = await client.send(outbound_request, stream=True)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Upstream request timed out while calling {upstream_url}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Upstream service unavailable while calling {upstream_url}: {exc}",
        ) from exc

    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=sanitize_response_headers(upstream_response.headers),
        background=BackgroundTask(upstream_response.aclose),
    )
