from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from services.legal_ai_service.assets.synthetic_dataset_source import RISK_LABELS

_RANDOM_STATE = 42


def main() -> None:
    args = _build_parser().parse_args()
    train_dataset = _read_dataset(args.training_csv)
    gold_dataset = _read_dataset(args.gold_csv)
    _train_and_export(
        train_dataset=train_dataset,
        gold_dataset=gold_dataset,
        artifacts_dir=args.artifacts_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the legal AI compliance risk baseline model.")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=default_root / "data" / "training.csv",
        help="Path to the exported training CSV",
    )
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=default_root / "data" / "gold.csv",
        help="Path to the exported gold CSV",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=default_root / "model_artifacts",
        help="Directory that will receive model files and reports",
    )
    return parser


def _train_and_export(
    *,
    train_dataset: list[dict[str, str]],
    gold_dataset: list[dict[str, str]],
    artifacts_dir: Path,
) -> None:
    try:
        import joblib
        import shap
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import StratifiedKFold
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Training requires scikit-learn, shap, and joblib to be installed"
        ) from exc

    train_texts = [row["text"] for row in train_dataset]
    train_labels = [row["risk_label"] for row in train_dataset]
    gold_texts = [row["text"] for row in gold_dataset]
    gold_labels = [row["risk_label"] for row in gold_dataset]

    def build_vectorizer() -> Any:
        preserved_tokens = {"all", "any", "no", "not", "without"}
        domain_generic_tokens = {
            "company",
            "employee",
            "employees",
            "hr",
            "leader",
            "leaders",
            "leadership",
            "manager",
            "managers",
            "management",
            "organization",
            "personnel",
            "policy",
            "staff",
            "supervisor",
            "supervisors",
            "team",
            "workplace",
        }
        stop_words = sorted((set(ENGLISH_STOP_WORDS) | domain_generic_tokens) - preserved_tokens)
        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words=stop_words,
            max_features=2500,
        )

    def build_classifier() -> Any:
        return LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            random_state=_RANDOM_STATE,
        )

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=_RANDOM_STATE)
    cv_true: list[str] = []
    cv_pred: list[str] = []
    for train_indices, validation_indices in splitter.split(train_texts, train_labels):
        fold_vectorizer = build_vectorizer()
        fold_classifier = build_classifier()
        fold_train_texts = [train_texts[index] for index in train_indices]
        fold_train_labels = [train_labels[index] for index in train_indices]
        fold_validation_texts = [train_texts[index] for index in validation_indices]
        fold_validation_labels = [train_labels[index] for index in validation_indices]
        fold_classifier.fit(
            fold_vectorizer.fit_transform(fold_train_texts),
            fold_train_labels,
        )
        cv_true.extend(fold_validation_labels)
        cv_pred.extend(fold_classifier.predict(fold_vectorizer.transform(fold_validation_texts)))

    vectorizer = build_vectorizer()
    classifier = build_classifier()
    train_matrix = vectorizer.fit_transform(train_texts)
    classifier.fit(train_matrix, train_labels)
    gold_predictions = classifier.predict(vectorizer.transform(gold_texts))

    background_texts = _select_background_texts(train_dataset, per_label=6)
    background_matrix = vectorizer.transform(background_texts)
    explainer = shap.LinearExplainer(classifier, background_matrix)
    training_explanations = explainer(train_matrix)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, artifacts_dir / "vectorizer.joblib")
    joblib.dump(classifier, artifacts_dir / "classifier.joblib")
    (artifacts_dir / "background_texts.json").write_text(
        json.dumps(background_texts, indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_type": "tfidf_logistic_regression",
                "dataset": {
                    "training_sample_count": len(train_dataset),
                    "gold_sample_count": len(gold_dataset),
                    "training_label_counts": _count_labels(train_labels),
                    "gold_label_counts": _count_labels(gold_labels),
                },
                "cross_validation": _build_metrics_payload(
                    y_true=cv_true,
                    y_pred=cv_pred,
                    accuracy_score=accuracy_score,
                    confusion_matrix=confusion_matrix,
                    f1_score=f1_score,
                    precision_recall_fscore_support=precision_recall_fscore_support,
                ),
                "gold_evaluation": _build_metrics_payload(
                    y_true=gold_labels,
                    y_pred=list(gold_predictions),
                    accuracy_score=accuracy_score,
                    confusion_matrix=confusion_matrix,
                    f1_score=f1_score,
                    precision_recall_fscore_support=precision_recall_fscore_support,
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "global_feature_importance.json").write_text(
        json.dumps(
            _build_global_feature_importance(
                feature_names=tuple(vectorizer.get_feature_names_out()),
                explanation_values=training_explanations.values,
                class_labels=tuple(classifier.classes_),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_dataset(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _count_labels(labels: list[str]) -> dict[str, int]:
    return {label: labels.count(label) for label in RISK_LABELS}


def _select_background_texts(dataset: list[dict[str, str]], *, per_label: int) -> list[str]:
    selected: list[str] = []
    for label in RISK_LABELS:
        label_rows = [row["text"] for row in dataset if row["risk_label"] == label]
        selected.extend(label_rows[:per_label])
    return selected


def _build_metrics_payload(
    *,
    y_true: list[str],
    y_pred: list[str],
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=RISK_LABELS,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=RISK_LABELS, average="macro")), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=RISK_LABELS).tolist(),
        "per_class_metrics": {
            label: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, label in enumerate(RISK_LABELS)
        },
    }


def _build_global_feature_importance(
    *,
    feature_names: tuple[str, ...],
    explanation_values: Any,
    class_labels: tuple[str, ...],
) -> dict[str, list[dict[str, float | str]]]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required to build the global SHAP report") from exc

    result: dict[str, list[dict[str, float | str]]] = {}
    for class_index, label in enumerate(class_labels):
        class_values = _extract_class_values(
            values=explanation_values,
            class_index=class_index,
            class_count=len(class_labels),
        )
        mean_abs = np.abs(class_values).mean(axis=0)
        sorted_indices = mean_abs.argsort()[::-1][:10]
        result[str(label)] = [
            {
                "term": feature_names[index],
                "mean_abs_shap": round(float(mean_abs[index]), 6),
            }
            for index in sorted_indices
        ]
    return result


def _extract_class_values(*, values: Any, class_index: int, class_count: int):
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required to build SHAP reports") from exc

    if isinstance(values, list):
        return np.asarray(values[class_index])
    ndarray = np.asarray(values)
    if ndarray.ndim == 2:
        return ndarray
    if ndarray.ndim == 3 and ndarray.shape[2] == class_count:
        return ndarray[:, :, class_index]
    if ndarray.ndim == 3 and ndarray.shape[1] == class_count:
        return ndarray[:, class_index, :]
    raise RuntimeError("Unsupported SHAP output shape while building the global report")


if __name__ == "__main__":
    main()
