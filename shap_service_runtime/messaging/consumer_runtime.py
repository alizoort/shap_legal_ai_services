from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from kombu import Producer

from ..config.common import get_common_settings
from .rabbitmq import (
    build_default_direct_exchange,
    build_retry_stage_queue,
    queue_name_for_key,
    retry_queue_name,
)

RETRY_ATTEMPT_HEADER = "x-retry-attempt"
ORIGINAL_QUEUE_HEADER = "x-original-queue"
LAST_ERROR_HEADER = "x-last-error"
LAST_ERROR_TYPE_HEADER = "x-last-error-type"
LAST_ERROR_AT_HEADER = "x-last-error-at"
MAX_ERROR_HEADER_LEN = 500


class ContractValidationError(ValueError):
    """Raised when event payload violates the messaging contract."""


@dataclass(frozen=True)
class RetryDecision:
    queue_key: str
    main_queue_name: str
    current_attempt: int
    next_attempt: int
    retry_queue_name: str
    next_retry_at: datetime


def get_retry_attempt(headers: Mapping[str, Any] | None) -> int:
    if not headers:
        return 0
    raw_value = headers.get(RETRY_ATTEMPT_HEADER)
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def build_error_context(
    exc: Exception,
    *,
    event_type: str | None = None,
    event_id: str | None = None,
) -> dict[str, str]:
    context = {
        "error_type": exc.__class__.__name__,
        "error": str(exc)[:MAX_ERROR_HEADER_LEN],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if event_type:
        context["event_type"] = event_type
    if event_id:
        context["event_id"] = event_id
    return context


def compute_retry_decision(
    *,
    queue_key: str,
    current_attempt: int,
    now: datetime | None = None,
) -> RetryDecision | None:
    settings = get_common_settings()
    if current_attempt < 0:
        raise ValueError("current_attempt must be >= 0")
    if current_attempt >= settings.rabbitmq_inbox_max_retries:
        return None

    next_attempt = current_attempt + 1
    main_queue_name = queue_name_for_key(queue_key)
    delay_ms = settings.rabbitmq_retry_delays_ms[current_attempt]
    now_ts = now or datetime.now(timezone.utc)
    return RetryDecision(
        queue_key=queue_key,
        main_queue_name=main_queue_name,
        current_attempt=current_attempt,
        next_attempt=next_attempt,
        retry_queue_name=retry_queue_name(main_queue_name, next_attempt),
        next_retry_at=now_ts + timedelta(milliseconds=delay_ms),
    )


def publish_to_retry_stage(
    *,
    message,  # noqa: ANN001
    body: dict,
    queue_key: str,
    current_attempt: int,
    error_context: Mapping[str, str] | None = None,
) -> RetryDecision:
    decision = compute_retry_decision(
        queue_key=queue_key,
        current_attempt=current_attempt,
    )
    if decision is None:
        raise RuntimeError("Retry decision not available; max retries exhausted")

    headers = dict(message.headers or {})
    headers[RETRY_ATTEMPT_HEADER] = decision.next_attempt
    headers[ORIGINAL_QUEUE_HEADER] = decision.main_queue_name
    if error_context:
        if "error" in error_context:
            headers[LAST_ERROR_HEADER] = str(error_context["error"])[:MAX_ERROR_HEADER_LEN]
        if "error_type" in error_context:
            headers[LAST_ERROR_TYPE_HEADER] = str(error_context["error_type"])[:100]
        if "timestamp" in error_context:
            headers[LAST_ERROR_AT_HEADER] = str(error_context["timestamp"])[:64]

    producer = Producer(message.channel)
    producer.publish(
        body,
        exchange=build_default_direct_exchange(),
        routing_key=decision.retry_queue_name,
        serializer="json",
        delivery_mode=2,
        message_id=message.properties.get("message_id"),
        correlation_id=message.properties.get("correlation_id"),
        headers=headers,
        declare=[build_retry_stage_queue(decision.main_queue_name, decision.next_attempt)],
        retry=True,
        retry_policy={
            "max_retries": 3,
            "interval_start": 0.2,
            "interval_step": 0.5,
            "interval_max": 2,
        },
    )
    return decision
