from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.legal_ai_service.scripts.export_synthetic_dataset import export_datasets
from services.legal_ai_service.scripts.train_compliance_model import (
    _build_global_feature_importance,
    _read_dataset,
    _train_and_export,
)

_MISSING_ML_DEPS = any(
    importlib.util.find_spec(package_name) is None
    for package_name in ("joblib", "numpy", "shap", "sklearn")
)


@unittest.skipIf(_MISSING_ML_DEPS, "Missing ML training dependencies")
class LegalAiTrainingPipelineTests(unittest.TestCase):
    def test_training_pipeline_exports_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data"
            artifacts_dir = root / "artifacts"

            export_datasets(data_dir)
            _train_and_export(
                train_dataset=_read_dataset(data_dir / "training.csv"),
                gold_dataset=_read_dataset(data_dir / "gold.csv"),
                artifacts_dir=artifacts_dir,
            )

            for relative_path in (
                "vectorizer.joblib",
                "classifier.joblib",
                "background_texts.json",
                "metrics.json",
                "global_feature_importance.json",
            ):
                self.assertTrue((artifacts_dir / relative_path).exists(), relative_path)

    def test_global_feature_importance_prefers_supportive_terms_over_generic_tokens(self) -> None:
        report = _build_global_feature_importance(
            feature_names=(
                "reserve",
                "all",
                "monitor",
                "without notice",
            ),
            explanation_values=[
                [
                    [0.05, 0.04, 0.01, 0.01],
                    [0.06, 0.05, 0.02, 0.02],
                ],
                [
                    [0.01, 0.01, 0.18, 0.22],
                    [0.01, 0.01, 0.16, 0.20],
                ],
                [
                    [0.02, 0.02, 0.03, 0.04],
                    [0.02, 0.02, 0.03, 0.04],
                ],
            ],
            class_labels=("low", "medium", "high"),
            sample_labels=("low", "low"),
        )

        self.assertEqual(
            [item["term"] for item in report["medium"][:2]],
            ["without notice", "monitor"],
        )


if __name__ == "__main__":
    unittest.main()
