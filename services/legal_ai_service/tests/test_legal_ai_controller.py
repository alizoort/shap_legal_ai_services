from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

os.environ.setdefault("ALLOW_DIRECT_PACKAGE_IMPORTS", "true")

from services.legal_ai_service.app.controllers.legal_ai import (
    LegalAiAnalyzeRequestModel,
    analyze,
    model_summary,
    ping,
)
from services.legal_ai_service.domain.entities import (
    AnalysisResult,
    ClassMetric,
    ClassProbabilities,
    EvaluationMetrics,
    FeatureContribution,
    GlobalFeatureImportance,
    GlobalFeatureImportanceItem,
    ModelSummary,
)
from services.legal_ai_service.domain.exceptions import ModelArtifactsUnavailableError


class _FakeAnalysisService:
    def analyze(self, *_args, **_kwargs):
        return AnalysisResult(
            risk_label="high",
            class_probabilities=ClassProbabilities(low=0.04, medium=0.14, high=0.82),
            needs_human_review=True,
            top_features_supporting_prediction=(
                FeatureContribution(term="without notice", shap_value=0.42),
                FeatureContribution(term="monitor", shap_value=0.39),
            ),
            top_features_against_prediction=(
                FeatureContribution(term="written notice", shap_value=-0.21),
            ),
            plain_english_explanation=(
                "This text is high risk because it suggests unrestricted monitoring."
            ),
            openai_commentary=None,
        )

    def get_model_summary(self):
        metrics = EvaluationMetrics(
            accuracy=0.81,
            macro_f1=0.8,
            confusion_matrix=((4, 1, 1), (1, 4, 1), (0, 1, 5)),
            per_class_metrics={
                "low": ClassMetric(precision=0.8, recall=0.7, f1=0.75, support=6),
                "medium": ClassMetric(precision=0.75, recall=0.8, f1=0.77, support=6),
                "high": ClassMetric(precision=0.83, recall=0.83, f1=0.83, support=6),
            },
        )
        return ModelSummary(
            model_type="tfidf_logistic_regression",
            training_sample_count=120,
            gold_sample_count=18,
            cross_validation=metrics,
            gold_evaluation=metrics,
            global_feature_importance=(
                GlobalFeatureImportance(
                    risk_label="high",
                    features=(
                        GlobalFeatureImportanceItem(term="without notice", mean_abs_shap=0.42),
                    ),
                ),
                GlobalFeatureImportance(
                    risk_label="medium",
                    features=(
                        GlobalFeatureImportanceItem(term="as needed", mean_abs_shap=0.31),
                    ),
                ),
                GlobalFeatureImportance(
                    risk_label="low",
                    features=(
                        GlobalFeatureImportanceItem(term="written notice", mean_abs_shap=0.28),
                    ),
                ),
            ),
        )


class LegalAiControllerTests(unittest.TestCase):
    def test_ping_endpoint_returns_reserved_routes(self) -> None:
        response = ping()

        self.assertEqual(response.service_name, "legal_ai_service")
        self.assertIn("/legal-ai/analyze", response.reserved_routes)

    @patch("services.legal_ai_service.app.controllers.legal_ai.get_legal_ai_analysis_service")
    def test_model_summary_endpoint_returns_metrics(self, mock_service_factory) -> None:
        mock_service_factory.return_value = _FakeAnalysisService()

        response = model_summary()

        self.assertEqual(response.model_type, "tfidf_logistic_regression")
        self.assertEqual(response.training_sample_count, 120)
        self.assertEqual(response.gold_sample_count, 18)

    @patch("services.legal_ai_service.app.controllers.legal_ai.get_legal_ai_analysis_service")
    def test_analyze_endpoint_returns_structured_result(self, mock_service_factory) -> None:
        mock_service_factory.return_value = _FakeAnalysisService()

        response = analyze(
            LegalAiAnalyzeRequestModel(
                text="We reserve the right to monitor all employee communications at any time without notice.",
                include_commentary=False,
            )
        )

        self.assertEqual(response.risk_label, "high")
        self.assertTrue(response.needs_human_review)
        self.assertIsNone(response.openai_commentary)
        self.assertEqual(response.top_features_supporting_prediction[0].term, "without notice")

    def test_analyze_endpoint_rejects_blank_text(self) -> None:
        with self.assertRaises(ValidationError):
            LegalAiAnalyzeRequestModel(text="   ", include_commentary=False)

    @patch("services.legal_ai_service.app.controllers.legal_ai.get_legal_ai_analysis_service")
    def test_model_summary_endpoint_returns_503_when_artifacts_are_missing(
        self,
        mock_service_factory,
    ) -> None:
        class _UnavailableService:
            def get_model_summary(self):
                raise ModelArtifactsUnavailableError("missing artifacts")

        mock_service_factory.return_value = _UnavailableService()

        with self.assertRaises(HTTPException) as context:
            model_summary()

        self.assertEqual(context.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
