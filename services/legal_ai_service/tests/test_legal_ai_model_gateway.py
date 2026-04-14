from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.legal_ai_service.domain.entities import FeatureContribution
from services.legal_ai_service.infrastructure.ml.legal_ai_model_gateway import (
    _build_plain_english_explanation,
    _extract_class_values,
    _select_display_contributions,
)


class LegalAiModelGatewayHelperTests(unittest.TestCase):
    def test_high_risk_explanation_uses_hr_compliance_themes(self) -> None:
        explanation = _build_plain_english_explanation(
            risk_label="high",
            supporting=(
                FeatureContribution(term="monitor", shap_value=0.41),
                FeatureContribution(term="without notice", shap_value=0.33),
            ),
            against=(
                FeatureContribution(term="written notice", shap_value=-0.12),
            ),
        )

        self.assertIn("high risk", explanation)
        self.assertIn("unrestricted monitoring", explanation)
        self.assertIn("lack of notice or consent", explanation)
        self.assertIn('"written notice"', explanation)

    def test_low_risk_explanation_prefers_positive_themes(self) -> None:
        explanation = _build_plain_english_explanation(
            risk_label="low",
            supporting=(
                FeatureContribution(term="written notice", shap_value=0.29),
                FeatureContribution(term="documented review", shap_value=0.21),
            ),
            against=(),
        )

        self.assertIn("low risk", explanation)
        self.assertIn("clear procedural safeguards", explanation)
        self.assertNotIn("lack of notice", explanation)

    def test_extract_class_values_supports_feature_major_multiclass_arrays(self) -> None:
        values = np.array(
            [
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ]
            ]
        )

        extracted = _extract_class_values(values=values, class_index=1, class_count=3)

        self.assertTrue(np.allclose(extracted, np.array([0.2, 0.5, 0.8])))

    def test_select_display_contributions_prefers_meaningful_terms(self) -> None:
        selected = _select_display_contributions(
            (
                FeatureContribution(term="reserve", shap_value=0.10),
                FeatureContribution(term="any", shap_value=0.09),
                FeatureContribution(term="all", shap_value=0.08),
                FeatureContribution(term="monitor", shap_value=0.07),
                FeatureContribution(term="without notice", shap_value=0.06),
            ),
            limit=3,
        )

        self.assertEqual(
            [item.term for item in selected],
            ["without notice", "monitor"],
        )


if __name__ == "__main__":
    unittest.main()
