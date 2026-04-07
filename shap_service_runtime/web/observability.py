from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..db.alembic import get_current_database_heads

ReadinessCheck = Callable[[], Awaitable[object] | object]

_REQUEST_COUNT = Counter(
    "shap_http_requests_total",
    "Count of handled HTTP requests.",
    ("service", "method", "route", "status_code"),
)
_REQUEST_DURATION = Histogram(
    "shap_http_request_duration_seconds",
    "Latency of handled HTTP requests.",
    ("service", "method", "route"),
)


def build_sqlalchemy_readiness_check(
    *,
    engine: Engine,
    dependency_name: str = "database",
) -> ReadinessCheck:
    def _check() -> dict[str, str]:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"dependency": dependency_name, "status": "ok"}

    return _check


def build_service_schema_readiness_check(
    *,
    engine: Engine,
    service_name: str,
    expected_heads: Sequence[str],
    dependency_name: str = "database",
    schema_dependency_name: str = "schema",
) -> ReadinessCheck:
    expected = tuple(sorted(str(head) for head in expected_heads if str(head).strip()))
    expected_heads_text = ", ".join(expected) or "<none>"

    def _check() -> list[dict[str, str]]:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            current_heads = get_current_database_heads(connection)

        if current_heads != expected:
            if not current_heads:
                raise RuntimeError(
                    f"{service_name} schema not ready: alembic version table is missing or empty; "
                    f"expected heads [{expected_heads_text}]"
                )

            current_heads_text = ", ".join(current_heads)
            raise RuntimeError(
                f"{service_name} schema not ready: expected heads [{expected_heads_text}], "
                f"found [{current_heads_text}]"
            )

        return [
            {"dependency": dependency_name, "status": "ok"},
            {"dependency": schema_dependency_name, "status": "ok"},
        ]

    return _check


def install_http_runtime_observability(
    app: FastAPI,
    *,
    service_name: str,
    readiness_check: ReadinessCheck | None = None,
) -> None:
    @app.middleware("http")
    async def _record_metrics(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        route_label = request.url.path
        try:
            response = await call_next(request)
            return response
        except Exception:
            route = request.scope.get("route")
            route_path = getattr(route, "path", None)
            if isinstance(route_path, str) and route_path:
                route_label = route_path
            duration = time.perf_counter() - start
            _REQUEST_COUNT.labels(service_name, request.method, route_label, "500").inc()
            _REQUEST_DURATION.labels(service_name, request.method, route_label).observe(duration)
            raise
        finally:
            route = request.scope.get("route")
            route_path = getattr(route, "path", None)
            if isinstance(route_path, str) and route_path:
                route_label = route_path
            if "response" in locals():
                duration = time.perf_counter() - start
                _REQUEST_COUNT.labels(
                    service_name,
                    request.method,
                    route_label,
                    str(response.status_code),
                ).inc()
                _REQUEST_DURATION.labels(service_name, request.method, route_label).observe(
                    duration
                )

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok", "service": service_name}

    @app.get("/ready", include_in_schema=False)
    async def ready() -> dict[str, object]:
        try:
            checks = await _run_readiness_check(readiness_check)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "service": service_name,
                    "reason": str(exc),
                },
            ) from exc

        return {
            "status": "ok",
            "service": service_name,
            "checks": checks,
        }

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _run_readiness_check(
    readiness_check: ReadinessCheck | None,
) -> list[dict[str, str]]:
    if readiness_check is None:
        return []

    result = readiness_check()
    if inspect.isawaitable(result):
        result = await result
    return _normalize_readiness_result(result)


def _normalize_readiness_result(result: object) -> list[dict[str, str]]:
    if result is None:
        return []
    if isinstance(result, dict):
        return [_normalize_check(result)]
    if isinstance(result, str):
        return [{"dependency": result, "status": "ok"}]
    if isinstance(result, Sequence):
        return [_normalize_check(item) for item in result]
    raise TypeError("Readiness check must return None, str, dict, or a sequence of dicts.")


def _normalize_check(item: object) -> dict[str, str]:
    if not isinstance(item, dict):
        raise TypeError("Readiness check sequence items must be dictionaries.")
    dependency = item.get("dependency", "dependency")
    status_value = item.get("status", "ok")
    return {
        "dependency": str(dependency),
        "status": str(status_value),
    }
