from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from services.legal_ai_service.config import settings
from services.legal_ai_service.domain.entities import (
    ClassMetric,
    ClassProbabilities,
    EvaluationMetrics,
    FeatureContribution,
    GlobalFeatureImportance,
    GlobalFeatureImportanceItem,
    ModelSummary,
    PredictionSnapshot,
    RISK_LABELS,
)
from services.legal_ai_service.domain.exceptions import ModelArtifactsUnavailableError

_TOP_FEATURE_COUNT = 5
_GLOBAL_FEATURE_COUNT = 8
_NEGATIVE_THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "unrestricted monitoring": (
        "monitor",
        "monitoring",
        "record",
        "communications",
        "keystrokes",
        "screen",
        "badge",
        "camera",
        "location",
        "device",
    ),
    "lack of notice or consent": (
        "notice",
        "consent",
        "approval",
        "retention",
        "access",
        "opt",
        "review",
    ),
    "unclear disciplinary process": (
        "appeal",
        "review path",
        "investigation",
        "disciplinary",
        "warning",
        "process",
        "documentation",
    ),
    "biased hiring or employment criteria": (
        "male",
        "female",
        "young",
        "age",
        "appearance",
        "culture",
        "fit",
        "native",
        "pregnancy",
        "religious",
        "marital",
    ),
    "unilateral pay changes": (
        "pay",
        "salary",
        "wage",
        "bonus",
        "overtime",
        "compensation",
        "deduct",
        "final pay",
    ),
    "employee data privacy concerns": (
        "data",
        "personal",
        "biometric",
        "vendor",
        "privacy",
        "records",
        "location",
        "contact",
    ),
}
_POSITIVE_THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "clear procedural safeguards": (
        "written",
        "documented",
        "notice",
        "approval",
        "appeal",
        "consent",
        "equal opportunity",
        "policy",
        "defined",
        "limited",
        "step",
        "timeline",
    ),
    "defined monitoring scope and transparency": (
        "monitor",
        "monitoring",
        "limited",
        "notice",
        "stated",
        "business hours",
        "specific purpose",
        "security review",
    ),
    "consent-based data usage": (
        "consent",
        "approval",
        "opt in",
        "retention",
        "delete",
        "request",
        "personal data",
        "access",
    ),
    "non-discriminatory employment criteria": (
        "equal opportunity",
        "qualification",
        "skills",
        "structured interview",
        "documented criteria",
        "job related",
    ),
    "documented compensation safeguards": (
        "written",
        "approval",
        "overtime",
        "payroll",
        "bonus",
        "salary",
        "compensation",
        "final pay",
    ),
}


@dataclass(frozen=True)
class _LoadedArtifacts:
    vectorizer: Any
    classifier: Any
    explainer: Any
    feature_names: tuple[str, ...]
    model_summary: ModelSummary


class SklearnLegalAiModelGateway:
    def __init__(self, artifacts_dir: str | Path | None = None) -> None:
        self._artifacts_dir = Path(artifacts_dir or settings.legal_ai_model_artifacts_dir)

    def analyze_text(self, text: str) -> PredictionSnapshot:
        if not text.strip():
            raise ValueError("text must not be empty")
        artifacts = _load_artifacts(str(self._artifacts_dir))
        vectorized_text = artifacts.vectorizer.transform([text])
        probabilities = artifacts.classifier.predict_proba(vectorized_text)[0]
        predicted_index = int(probabilities.argmax())
        predicted_label = str(artifacts.classifier.classes_[predicted_index])
        explanation = artifacts.explainer(vectorized_text)
        shap_values = _extract_class_values(
            values=explanation.values,
            class_index=predicted_index,
            class_count=len(artifacts.classifier.classes_),
        )
        present_indices = list(vectorized_text.nonzero()[1])
        contributions = [
            FeatureContribution(
                term=artifacts.feature_names[index],
                shap_value=round(float(shap_values[index]), 6),
            )
            for index in present_indices
        ]
        supporting = tuple(
            sorted(
                (item for item in contributions if item.shap_value > 0),
                key=lambda item: item.shap_value,
                reverse=True,
            )[:_TOP_FEATURE_COUNT]
        )
        against = tuple(
            sorted(
                (item for item in contributions if item.shap_value < 0),
                key=lambda item: item.shap_value,
            )[:_TOP_FEATURE_COUNT]
        )
        if not supporting:
            supporting = tuple(
                sorted(contributions, key=lambda item: item.shap_value, reverse=True)[:_TOP_FEATURE_COUNT]
            )
        return PredictionSnapshot(
            risk_label=predicted_label,
            class_probabilities=_build_class_probabilities(
                class_labels=artifacts.classifier.classes_,
                probabilities=probabilities,
            ),
            top_features_supporting_prediction=supporting,
            top_features_against_prediction=against,
            plain_english_explanation=_build_plain_english_explanation(
                risk_label=predicted_label,
                supporting=supporting,
                against=against,
            ),
        )

    def get_model_summary(self) -> ModelSummary:
        return _load_artifacts(str(self._artifacts_dir)).model_summary


@lru_cache(maxsize=4)
def _load_artifacts(artifacts_dir: str) -> _LoadedArtifacts:
    path = Path(artifacts_dir)
    required = {
        "vectorizer": path / "vectorizer.joblib",
        "classifier": path / "classifier.joblib",
        "background": path / "background_texts.json",
        "metrics": path / "metrics.json",
        "global_importance": path / "global_feature_importance.json",
    }
    missing = [name for name, candidate in required.items() if not candidate.exists()]
    if missing:
        raise ModelArtifactsUnavailableError(
            f"Legal AI model artifacts are missing from {path}: {', '.join(sorted(missing))}"
        )

    try:
        import joblib
        import shap
    except ModuleNotFoundError as exc:
        raise ModelArtifactsUnavailableError(
            "ML dependencies are not installed for legal_ai_service"
        ) from exc

    vectorizer = joblib.load(required["vectorizer"])
    classifier = joblib.load(required["classifier"])
    background_texts = json.loads(required["background"].read_text(encoding="utf-8"))
    background_matrix = vectorizer.transform(background_texts)
    explainer = shap.LinearExplainer(classifier, background_matrix)
    metrics_payload = json.loads(required["metrics"].read_text(encoding="utf-8"))
    global_importance_payload = json.loads(required["global_importance"].read_text(encoding="utf-8"))
    return _LoadedArtifacts(
        vectorizer=vectorizer,
        classifier=classifier,
        explainer=explainer,
        feature_names=tuple(vectorizer.get_feature_names_out()),
        model_summary=_build_model_summary(metrics_payload, global_importance_payload),
    )


def _build_model_summary(
    metrics_payload: dict[str, Any],
    global_importance_payload: dict[str, Any],
) -> ModelSummary:
    return ModelSummary(
        model_type=str(metrics_payload["model_type"]),
        training_sample_count=int(metrics_payload["dataset"]["training_sample_count"]),
        gold_sample_count=int(metrics_payload["dataset"]["gold_sample_count"]),
        cross_validation=_build_evaluation_metrics(metrics_payload["cross_validation"]),
        gold_evaluation=_build_evaluation_metrics(metrics_payload["gold_evaluation"]),
        global_feature_importance=tuple(
            GlobalFeatureImportance(
                risk_label=label,
                features=tuple(
                    GlobalFeatureImportanceItem(
                        term=str(item["term"]),
                        mean_abs_shap=round(float(item["mean_abs_shap"]), 6),
                    )
                    for item in global_importance_payload.get(label, [])[:_GLOBAL_FEATURE_COUNT]
                ),
            )
            for label in RISK_LABELS
        ),
    )


def _build_evaluation_metrics(raw: dict[str, Any]) -> EvaluationMetrics:
    return EvaluationMetrics(
        accuracy=round(float(raw["accuracy"]), 6),
        macro_f1=round(float(raw["macro_f1"]), 6),
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in raw["confusion_matrix"]),
        per_class_metrics={
            label: ClassMetric(
                precision=round(float(metrics["precision"]), 6),
                recall=round(float(metrics["recall"]), 6),
                f1=round(float(metrics["f1"]), 6),
                support=int(metrics["support"]),
            )
            for label, metrics in raw["per_class_metrics"].items()
        },
    )


def _build_class_probabilities(class_labels: Iterable[str], probabilities: Any) -> ClassProbabilities:
    mapping = {str(label): float(probability) for label, probability in zip(class_labels, probabilities)}
    return ClassProbabilities(
        low=round(mapping.get("low", 0.0), 6),
        medium=round(mapping.get("medium", 0.0), 6),
        high=round(mapping.get("high", 0.0), 6),
    )


def _build_plain_english_explanation(
    *,
    risk_label: str,
    supporting: tuple[FeatureContribution, ...],
    against: tuple[FeatureContribution, ...],
) -> str:
    supporting_terms = _dedupe_terms([item.term for item in supporting if item.shap_value > 0])
    if not supporting_terms:
        supporting_terms = _dedupe_terms([item.term for item in supporting])
    against_terms = _dedupe_terms([item.term for item in against if item.shap_value < 0])
    themes = _derive_themes(supporting_terms, risk_label=risk_label)
    supporting_phrase = _summarize_feature_phrases(supporting_terms[:2])
    if risk_label == "high":
        if len(themes) > 1:
            message = f"This text is high risk because it suggests {themes[0]} and {themes[1]}."
        elif themes:
            message = (
                f"This text is high risk because it suggests {themes[0]} "
                f"and uses terms like {supporting_phrase}."
            )
        else:
            message = (
                f"This text is high risk because terms like {supporting_phrase} "
                "indicate elevated employment compliance risk."
            )
    elif risk_label == "medium":
        if len(themes) > 1:
            message = (
                f"This text is medium risk because it raises concerns about {themes[0]} "
                f"while leaving {themes[1]} unresolved."
            )
        elif themes:
            message = (
                f"This text is medium risk because it raises concerns about {themes[0]} "
                "but does not fully clarify the safeguards."
            )
        else:
            message = (
                f"This text is medium risk because terms like {supporting_phrase} "
                "leave important compliance safeguards unclear."
            )
    else:
        if len(themes) > 1:
            message = f"This text is low risk because it includes {themes[0]} and {themes[1]}."
        elif themes:
            message = (
                f"This text is low risk because it includes {themes[0]} "
                "with clearly stated safeguards."
            )
        else:
            message = (
                f"This text is low risk because terms like {supporting_phrase} "
                "show clearer HR and compliance safeguards."
            )
    if against_terms:
        message += (
            " Counter-signals such as "
            f"{_summarize_feature_phrases(against_terms[:2])} were present but did not drive the prediction."
        )
    return message


def _derive_themes(terms: list[str], *, risk_label: str) -> list[str]:
    theme_keywords = _POSITIVE_THEME_KEYWORDS if risk_label == "low" else _NEGATIVE_THEME_KEYWORDS
    scores: defaultdict[str, int] = defaultdict(int)
    for term in terms:
        lowered = term.lower()
        for theme, keywords in theme_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                scores[theme] += 1
    return [theme for theme, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _summarize_feature_phrases(terms: list[str]) -> str:
    selected = [term.replace("_", " ").strip() for term in terms if term]
    if not selected:
        return "the wording used in the policy"
    if len(selected) == 1:
        return f'"{selected[0]}"'
    return f'"{selected[0]}" and "{selected[1]}"'


def _dedupe_terms(terms: list[str]) -> list[str]:
    ordered_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered_terms.append(term)
    return ordered_terms


def _extract_class_values(*, values: Any, class_index: int, class_count: int):
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModelArtifactsUnavailableError(
            "NumPy is required to extract SHAP values for legal_ai_service"
        ) from exc

    if isinstance(values, list):
        return np.asarray(values[class_index])[0]
    ndarray = np.asarray(values)
    if ndarray.ndim == 2:
        return ndarray[0]
    if ndarray.ndim == 3 and ndarray.shape[2] == class_count:
        return ndarray[0, :, class_index]
    if ndarray.ndim == 3 and ndarray.shape[1] == class_count:
        return ndarray[0, class_index, :]
    raise ModelArtifactsUnavailableError("Unsupported SHAP output shape for multiclass classifier")
