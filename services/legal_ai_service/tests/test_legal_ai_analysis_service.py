from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.legal_ai_service.application.dtos.legal_ai import LegalAiAnalyzeRequest
from services.legal_ai_service.application.services.legal_ai_analysis_service import (
    LegalAiAnalysisService,
)
from services.legal_ai_service.domain.entities import (
    ClassProbabilities,
    FeatureContribution,
    OpenAiCommentary,
    PredictionSnapshot,
)


class _FakeModelGateway:
    def __init__(self, prediction: PredictionSnapshot) -> None:
        self.prediction = prediction

    def analyze_text(self, text: str) -> PredictionSnapshot:
        self.last_text = text
        return self.prediction

    def get_model_summary(self):  # pragma: no cover - not exercised here
        raise NotImplementedError


class _FakeCommentaryGateway:
    def __init__(self, commentary: OpenAiCommentary | None) -> None:
        self.commentary = commentary
        self.call_count = 0

    def build_commentary(self, _analysis_result):
        self.call_count += 1
        return self.commentary


class LegalAiAnalysisServiceTests(unittest.TestCase):
    def test_low_risk_with_low_confidence_still_requires_human_review(self) -> None:
        service = LegalAiAnalysisService(
            model_gateway=_FakeModelGateway(
                PredictionSnapshot(
                    risk_label="low",
                    class_probabilities=ClassProbabilities(low=0.55, medium=0.25, high=0.20),
                    top_features_supporting_prediction=(
                        FeatureContribution(term="written notice", shap_value=0.22),
                    ),
                    top_features_against_prediction=(),
                    plain_english_explanation="This text is low risk because it includes written notice.",
                )
            ),
            commentary_gateway=_FakeCommentaryGateway(commentary=None),
        )

        result = service.analyze(
            LegalAiAnalyzeRequest(
                text="Written notice is provided before changes are made.",
                include_commentary=False,
            )
        )

        self.assertTrue(result.needs_human_review)
        self.assertIsNone(result.openai_commentary)

    def test_commentary_is_only_requested_when_enabled(self) -> None:
        commentary_gateway = _FakeCommentaryGateway(
            commentary=OpenAiCommentary(
                summary="HR should review the monitoring scope.",
                key_concerns=("scope is broad",),
                recommended_next_steps=("limit monitoring to stated purposes",),
                disclaimer="POC only. Not legal advice.",
            )
        )
        service = LegalAiAnalysisService(
            model_gateway=_FakeModelGateway(
                PredictionSnapshot(
                    risk_label="high",
                    class_probabilities=ClassProbabilities(low=0.05, medium=0.15, high=0.80),
                    top_features_supporting_prediction=(
                        FeatureContribution(term="without notice", shap_value=0.42),
                    ),
                    top_features_against_prediction=(),
                    plain_english_explanation="This text is high risk because it suggests unrestricted monitoring.",
                )
            ),
            commentary_gateway=commentary_gateway,
        )

        result = service.analyze(
            LegalAiAnalyzeRequest(
                text="We may monitor all employee communications without notice.",
                include_commentary=True,
            )
        )

        self.assertEqual(commentary_gateway.call_count, 1)
        self.assertIsNotNone(result.openai_commentary)


if __name__ == "__main__":
    unittest.main()
