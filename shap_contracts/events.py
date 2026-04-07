from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, Literal, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


T = TypeVar("T", bound=BaseModel)

EventType = Literal["PlaceholderReserved"]

EVENT_ROUTING_KEYS: dict[EventType, str] = {
    "PlaceholderReserved": "legal_ai.placeholder_reserved",
}


class EventEnvelope(BaseModel, Generic[T]):
    event_id: UUID = Field(default_factory=uuid4)
    event_version: int = 1
    event_type: EventType
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    producer: str
    correlation_id: str | None = None
    payload: T


class PlaceholderReservedPayload(BaseModel):
    placeholder_key: str
    detail: str


def event_type_to_routing_key(event_type: str) -> str:
    try:
        return EVENT_ROUTING_KEYS[event_type]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"Unsupported event type: {event_type}") from exc
