from __future__ import annotations

import random
import socket
import time
from collections.abc import Callable, Sequence
from typing import Any

from kombu import Consumer
from kombu.exceptions import OperationalError as KombuOperationalError

from ..config.common import get_common_settings

RECOVERABLE_CONNECTION_EXCEPTIONS = (
    ConnectionResetError,
    BrokenPipeError,
    OSError,
    KombuOperationalError,
)


def _compute_reconnect_delay(
    *,
    attempt: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> float:
    if attempt <= 0:
        raise ValueError("attempt must be >= 1")

    exponent = min(attempt - 1, 10)
    capped = min(base_delay_seconds * (2**exponent), max_delay_seconds)
    jitter = capped * 0.2
    with_jitter = capped + random.uniform(-jitter, jitter)
    return max(base_delay_seconds, min(max_delay_seconds, with_jitter))


def run_kombu_consumer_forever(
    *,
    logger,
    consumer_label: str,
    queue,
    callback: Callable[[dict[str, Any], Any], None],
    connection_factory: Callable[[], Any],
    accept: Sequence[str] = ("json",),
    prefetch_count: int | None = None,
    poll_interval_seconds: float | None = None,
    heartbeat_seconds: int | None = None,
) -> None:
    settings = get_common_settings()
    poll_interval = max(
        float(
            poll_interval_seconds
            if poll_interval_seconds is not None
            else settings.rabbitmq_poll_interval_seconds
        ),
        0.1,
    )
    heartbeat_timeout = max(
        int(
            heartbeat_seconds
            if heartbeat_seconds is not None
            else settings.rabbitmq_heartbeat_seconds
        ),
        1,
    )
    base_delay = max(poll_interval, 1.0)
    max_delay = max(base_delay, float(max(heartbeat_timeout * 2, 10)))
    stable_connection_seconds = float(max(heartbeat_timeout * 2, 30))
    reconnect_attempt = 0
    connected_at: float | None = None

    while True:
        try:
            with connection_factory() as connection:
                connected_at = time.monotonic()
                with Consumer(
                    connection,
                    queues=[queue],
                    callbacks=[callback],
                    accept=list(accept),
                    prefetch_count=prefetch_count,
                ):
                    while True:
                        try:
                            connection.drain_events(timeout=poll_interval)
                        except socket.timeout:
                            connection.heartbeat_check()
                            continue
        except RECOVERABLE_CONNECTION_EXCEPTIONS as exc:
            uptime_seconds = (
                time.monotonic() - connected_at if connected_at is not None else 0.0
            )
            if uptime_seconds >= stable_connection_seconds:
                reconnect_attempt = 1
            else:
                reconnect_attempt += 1
            reconnect_delay = _compute_reconnect_delay(
                attempt=reconnect_attempt,
                base_delay_seconds=base_delay,
                max_delay_seconds=max_delay,
            )
            logger.warning(
                "%s lost RabbitMQ connection; retrying in %.1fs: %s",
                consumer_label,
                reconnect_delay,
                exc,
            )
            time.sleep(reconnect_delay)
