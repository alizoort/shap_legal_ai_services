from __future__ import annotations

from dataclasses import dataclass

RISK_LABELS: tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class FeatureContribution:
    term: str
    shap_value: float


@dataclass(frozen=True)
class GlobalFeatureImportanceItem:
    term: str
    mean_abs_shap: float


@dataclass(frozen=True)
class OpenAiCommentary:
    summary: str
    key_concerns: tuple[str, ...]
    recommended_next_steps: tuple[str, ...]
    disclaimer: str


@dataclass(frozen=True)
class ClassProbabilities:
    low: float
    medium: float
    high: float

    def max_probability(self) -> float:
        return max(self.low, self.medium, self.high)


@dataclass(frozen=True)
class PredictionSnapshot:
    risk_label: str
    class_probabilities: ClassProbabilities
    top_features_supporting_prediction: tuple[FeatureContribution, ...]
    top_features_against_prediction: tuple[FeatureContribution, ...]
    plain_english_explanation: str


@dataclass(frozen=True)
class AnalysisResult:
    risk_label: str
    class_probabilities: ClassProbabilities
    needs_human_review: bool
    top_features_supporting_prediction: tuple[FeatureContribution, ...]
    top_features_against_prediction: tuple[FeatureContribution, ...]
    plain_english_explanation: str
    openai_commentary: OpenAiCommentary | None


@dataclass(frozen=True)
class ClassMetric:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    macro_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class_metrics: dict[str, ClassMetric]


@dataclass(frozen=True)
class GlobalFeatureImportance:
    risk_label: str
    features: tuple[GlobalFeatureImportanceItem, ...]


@dataclass(frozen=True)
class ModelSummary:
    model_type: str
    training_sample_count: int
    gold_sample_count: int
    cross_validation: EvaluationMetrics
    gold_evaluation: EvaluationMetrics
    global_feature_importance: tuple[GlobalFeatureImportance, ...]

