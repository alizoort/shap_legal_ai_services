from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from services.legal_ai_service.config import settings
from services.legal_ai_service.domain.entities import AnalysisResult, OpenAiCommentary

_COMMENTARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "key_concerns": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
        "recommended_next_steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
        "disclaimer": {"type": "string"},
    },
    "required": [
        "summary",
        "key_concerns",
        "recommended_next_steps",
        "disclaimer",
    ],
}


class NullCommentaryGateway:
    def build_commentary(self, analysis_result: AnalysisResult) -> OpenAiCommentary | None:
        _ = analysis_result
        return None


class OpenAiCommentaryGateway:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def build_commentary(self, analysis_result: AnalysisResult) -> OpenAiCommentary | None:
        if not self._api_key.strip():
            return None
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            return None

        client = OpenAI(api_key=self._api_key)
        try:
            response = client.responses.create(
                model=self._model,
                instructions=(
                    "You are generating structured HR/compliance commentary for a demo system. "
                    "Use only the supplied classifier output. Do not reclassify the text. "
                    "Keep the tone professional, concise, and suitable for non-technical stakeholders."
                ),
                input=json.dumps(_build_commentary_payload(analysis_result), ensure_ascii=True),
                max_output_tokens=400,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "legal_ai_commentary",
                        "strict": True,
                        "schema": _COMMENTARY_SCHEMA,
                    }
                },
            )
            payload = json.loads(response.output_text)
        except Exception:
            return None

        return OpenAiCommentary(
            summary=str(payload["summary"]),
            key_concerns=tuple(str(item) for item in payload["key_concerns"]),
            recommended_next_steps=tuple(str(item) for item in payload["recommended_next_steps"]),
            disclaimer=str(payload["disclaimer"]),
        )


def build_commentary_gateway() -> OpenAiCommentaryGateway | NullCommentaryGateway:
    if not settings.legal_ai_enable_commentary:
        return NullCommentaryGateway()
    if settings.llm_provider.lower() != "openai":
        return NullCommentaryGateway()
    return OpenAiCommentaryGateway(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


def _build_commentary_payload(analysis_result: AnalysisResult) -> dict[str, Any]:
    payload = asdict(analysis_result)
    payload.pop("openai_commentary", None)
    return payload

