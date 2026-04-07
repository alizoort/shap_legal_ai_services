from __future__ import annotations

from kombu import Connection, Exchange, Queue

from shap_contracts.events import (
    EVENT_ROUTING_KEYS,
    event_type_to_routing_key as contract_event_type_to_routing_key,
)

from ..config.common import get_common_settings


def _queue_key_to_name() -> dict[str, str]:
    settings = get_common_settings()
    return {
        "legal_ai_events": settings.rabbitmq_legal_ai_events_queue,
    }


def event_type_to_routing_key(event_type: str) -> str:
    return contract_event_type_to_routing_key(event_type)


def build_connection() -> Connection:
    settings = get_common_settings()
    return Connection(
        settings.rabbitmq_url,
        heartbeat=settings.rabbitmq_heartbeat_seconds,
        connect_timeout=settings.rabbitmq_connect_timeout_seconds,
    )


def build_event_exchange() -> Exchange:
    settings = get_common_settings()
    return Exchange(settings.rabbitmq_exchange, type="topic", durable=True)


def build_default_direct_exchange() -> Exchange:
    return Exchange("", type="direct", durable=True, no_declare=True)


def queue_name_for_key(queue_key: str) -> str:
    queue_key_to_name = _queue_key_to_name()
    try:
        return queue_key_to_name[queue_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported queue key: {queue_key}") from exc


def dead_letter_queue_name(main_queue_name: str) -> str:
    settings = get_common_settings()
    return f"{main_queue_name}{settings.rabbitmq_dlq_suffix}"


def retry_queue_name(main_queue_name: str, attempt: int) -> str:
    if attempt <= 0:
        raise ValueError("attempt must be >= 1")
    settings = get_common_settings()
    return f"{main_queue_name}{settings.rabbitmq_retry_suffix}.{attempt}"


def retry_delay_ms_for_attempt(attempt: int) -> int:
    if attempt <= 0:
        raise ValueError("attempt must be >= 1")
    settings = get_common_settings()
    try:
        return settings.rabbitmq_retry_delays_ms[attempt - 1]
    except IndexError as exc:
        raise ValueError(f"No configured retry delay for attempt {attempt}") from exc


def build_dead_letter_queue(main_queue_name: str) -> Queue:
    return Queue(
        dead_letter_queue_name(main_queue_name),
        exchange=build_default_direct_exchange(),
        routing_key=dead_letter_queue_name(main_queue_name),
        durable=True,
    )


def build_retry_stage_queue(main_queue_name: str, attempt: int) -> Queue:
    return Queue(
        retry_queue_name(main_queue_name, attempt),
        exchange=build_default_direct_exchange(),
        routing_key=retry_queue_name(main_queue_name, attempt),
        durable=True,
        queue_arguments={
            "x-message-ttl": retry_delay_ms_for_attempt(attempt),
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": main_queue_name,
        },
    )


def _build_primary_queue(
    *,
    queue_name: str,
    exchange: Exchange,
    routing_key: str,
) -> Queue:
    return Queue(
        queue_name,
        exchange=exchange,
        routing_key=routing_key,
        durable=True,
        queue_arguments={
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": dead_letter_queue_name(queue_name),
        },
    )


def build_event_queues(exchange: Exchange | None = None) -> dict[str, Queue]:
    settings = get_common_settings()
    queue_key_to_name = _queue_key_to_name()
    event_exchange = exchange or build_event_exchange()
    queues: dict[str, Queue] = {
        "legal_ai_events": _build_primary_queue(
            queue_name=settings.rabbitmq_legal_ai_events_queue,
            exchange=event_exchange,
            routing_key="legal_ai.*",
        ),
    }

    for queue_key, main_name in queue_key_to_name.items():
        queues[f"{queue_key}_dlq"] = build_dead_letter_queue(main_name)
        for attempt in range(1, settings.rabbitmq_inbox_max_retries + 1):
            queues[f"{queue_key}_retry_{attempt}"] = build_retry_stage_queue(
                main_name,
                attempt,
            )
    return queues
