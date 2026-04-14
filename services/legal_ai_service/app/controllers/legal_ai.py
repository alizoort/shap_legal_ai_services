from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.legal_ai_service.app.dependencies import get_legal_ai_analysis_service
from services.legal_ai_service.application.dtos.legal_ai import LegalAiAnalyzeRequest
from services.legal_ai_service.application.services.legal_ai_scaffold_service import (
    LegalAiScaffoldService,
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
    OpenAiCommentary,
)
from services.legal_ai_service.domain.exceptions import ModelArtifactsUnavailableError

router = APIRouter(prefix="/legal-ai", tags=["legal-ai"])


class LegalAiPingResponseModel(BaseModel):
    service_name: str
    status: str
    message: str
    reserved_routes: list[str]


class LegalAiAnalyzeRequestModel(BaseModel):
    text: str = Field(..., min_length=1)
    include_commentary: bool = False

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class ClassProbabilitiesModel(BaseModel):
    low: float
    medium: float
    high: float


class FeatureContributionModel(BaseModel):
    term: str
    shap_value: float


class OpenAiCommentaryModel(BaseModel):
    summary: str
    key_concerns: list[str]
    recommended_next_steps: list[str]
    disclaimer: str


class LegalAiAnalyzeResponseModel(BaseModel):
    risk_label: str
    class_probabilities: ClassProbabilitiesModel
    needs_human_review: bool
    top_features_supporting_prediction: list[FeatureContributionModel]
    top_features_against_prediction: list[FeatureContributionModel]
    plain_english_explanation: str
    openai_commentary: OpenAiCommentaryModel | None


class ClassMetricModel(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class EvaluationMetricsModel(BaseModel):
    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    per_class_metrics: dict[str, ClassMetricModel]


class GlobalFeatureImportanceItemModel(BaseModel):
    term: str
    mean_abs_shap: float


class GlobalFeatureImportanceModel(BaseModel):
    risk_label: str
    features: list[GlobalFeatureImportanceItemModel]


class LegalAiModelSummaryResponseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    training_sample_count: int
    gold_sample_count: int
    cross_validation: EvaluationMetricsModel
    gold_evaluation: EvaluationMetricsModel
    global_feature_importance: list[GlobalFeatureImportanceModel]


@router.get("/ping", response_model=LegalAiPingResponseModel)
def ping() -> LegalAiPingResponseModel:
    response = LegalAiScaffoldService().build_ping_response()
    return LegalAiPingResponseModel(
        service_name=response.service_name,
        status=response.status,
        message=response.message,
        reserved_routes=list(response.reserved_routes),
    )


@router.get("/model-summary", response_model=LegalAiModelSummaryResponseModel)
def model_summary() -> LegalAiModelSummaryResponseModel:
    try:
        summary = get_legal_ai_analysis_service().get_model_summary()
    except ModelArtifactsUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return _to_model_summary_response(summary)


@router.post("/analyze", response_model=LegalAiAnalyzeResponseModel)
def analyze(request: LegalAiAnalyzeRequestModel) -> LegalAiAnalyzeResponseModel:
    try:
        result = get_legal_ai_analysis_service().analyze(
            LegalAiAnalyzeRequest(
                text=request.text,
                include_commentary=request.include_commentary,
            )
        )
    except ModelArtifactsUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return _to_analysis_response(result)


def _to_analysis_response(result: AnalysisResult) -> LegalAiAnalyzeResponseModel:
    return LegalAiAnalyzeResponseModel(
        risk_label=result.risk_label,
        class_probabilities=_to_class_probabilities(result.class_probabilities),
        needs_human_review=result.needs_human_review,
        top_features_supporting_prediction=_to_feature_contributions(
            result.top_features_supporting_prediction
        ),
        top_features_against_prediction=_to_feature_contributions(
            result.top_features_against_prediction
        ),
        plain_english_explanation=result.plain_english_explanation,
        openai_commentary=_to_openai_commentary(result.openai_commentary),
    )


def _to_model_summary_response(summary: ModelSummary) -> LegalAiModelSummaryResponseModel:
    return LegalAiModelSummaryResponseModel(
        model_type=summary.model_type,
        training_sample_count=summary.training_sample_count,
        gold_sample_count=summary.gold_sample_count,
        cross_validation=_to_evaluation_metrics(summary.cross_validation),
        gold_evaluation=_to_evaluation_metrics(summary.gold_evaluation),
        global_feature_importance=[
            GlobalFeatureImportanceModel(
                risk_label=item.risk_label,
                features=[
                    GlobalFeatureImportanceItemModel(
                        term=feature.term,
                        mean_abs_shap=feature.mean_abs_shap,
                    )
                    for feature in item.features
                ],
            )
            for item in summary.global_feature_importance
        ],
    )


def _to_class_probabilities(value: ClassProbabilities) -> ClassProbabilitiesModel:
    return ClassProbabilitiesModel(
        low=value.low,
        medium=value.medium,
        high=value.high,
    )


def _to_feature_contributions(
    items: tuple[FeatureContribution, ...],
) -> list[FeatureContributionModel]:
    return [
        FeatureContributionModel(term=item.term, shap_value=item.shap_value)
        for item in items
    ]


def _to_openai_commentary(value: OpenAiCommentary | None) -> OpenAiCommentaryModel | None:
    if value is None:
        return None
    return OpenAiCommentaryModel(
        summary=value.summary,
        key_concerns=list(value.key_concerns),
        recommended_next_steps=list(value.recommended_next_steps),
        disclaimer=value.disclaimer,
    )


def _to_evaluation_metrics(value: EvaluationMetrics) -> EvaluationMetricsModel:
    return EvaluationMetricsModel(
        accuracy=value.accuracy,
        macro_f1=value.macro_f1,
        confusion_matrix=[list(row) for row in value.confusion_matrix],
        per_class_metrics={
            label: _to_class_metric(metric)
            for label, metric in value.per_class_metrics.items()
        },
    )


def _to_class_metric(value: ClassMetric) -> ClassMetricModel:
    return ClassMetricModel(
        precision=value.precision,
        recall=value.recall,
        f1=value.f1,
        support=value.support,
    )
