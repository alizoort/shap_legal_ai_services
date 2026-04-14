from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalAiPingResponse:
    service_name: str
    status: str
    message: str
    reserved_routes: tuple[str, ...]


@dataclass(frozen=True)
class LegalAiAnalyzeRequest:
    text: str
    include_commentary: bool = False
