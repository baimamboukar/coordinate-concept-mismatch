from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def select_balanced_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    index = int(np.argmax(true_positive_rate - false_positive_rate))
    threshold = float(thresholds[index])
    if np.isfinite(threshold):
        return threshold
    return float(np.nextafter(np.max(scores), np.inf))


def select_fpr_thresholds(
    labels: np.ndarray, scores: np.ndarray, target_fprs: Iterable[float]
) -> dict[float, float]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    selected = {}
    for target in target_fprs:
        valid = np.flatnonzero(false_positive_rate <= target)
        if not len(valid):
            raise ValueError(f"No ROC operating point is available at FPR {target}.")
        best = valid[np.argmax(true_positive_rate[valid])]
        threshold = float(thresholds[best])
        if not np.isfinite(threshold):
            threshold = float(np.nextafter(np.max(scores), np.inf))
        selected[float(target)] = threshold
    return selected


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    target_fprs: Iterable[float] = (0.01, 0.05),
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    probabilities = _sigmoid(scores)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)

    metrics: dict[str, float | int] = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
        "expected_calibration_error": _expected_calibration_error(labels, probabilities),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    for target in target_fprs:
        valid = np.flatnonzero(false_positive_rate <= target)
        value = float(np.max(true_positive_rate[valid])) if len(valid) else 0.0
        metrics[f"tpr_at_fpr_{_percentage_name(target)}"] = value
    return metrics


def fixed_operating_point_metrics(
    labels: np.ndarray, scores: np.ndarray, thresholds: dict[float, float]
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    metrics = {}
    negative_count = max(int(np.sum(labels == 0)), 1)
    positive_count = max(int(np.sum(labels == 1)), 1)
    for target, threshold in thresholds.items():
        predictions = scores >= threshold
        fp = int(np.sum(predictions & (labels == 0)))
        tp = int(np.sum(predictions & (labels == 1)))
        suffix = _percentage_name(target)
        metrics[f"achieved_fpr_at_source_{suffix}"] = fp / negative_count
        metrics[f"tpr_at_source_{suffix}"] = tp / positive_count
        metrics[f"source_threshold_{suffix}"] = float(threshold)
    return metrics


def paired_auroc_gap_interval(
    labels: np.ndarray,
    oracle_scores: np.ndarray,
    transfer_scores: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    oracle_scores = np.asarray(oracle_scores, dtype=np.float64)
    transfer_scores = np.asarray(transfer_scores, dtype=np.float64)
    if not (len(labels) == len(oracle_scores) == len(transfer_scores)):
        raise ValueError("Labels and paired score arrays must have equal lengths.")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("Bootstrap samples and confidence must be valid.")

    observed = float(roc_auc_score(labels, oracle_scores) - roc_auc_score(labels, transfer_scores))
    rng = np.random.default_rng(seed)
    differences = []
    while len(differences) < samples:
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        differences.append(
            roc_auc_score(sampled_labels, oracle_scores[indices])
            - roc_auc_score(sampled_labels, transfer_scores[indices])
        )

    tail = (1 - confidence) / 2
    lower, upper = np.quantile(differences, [tail, 1 - tail])
    return observed, float(lower), float(upper)


def prediction_rows(
    row_ids: Sequence[Any], labels: np.ndarray, scores: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    probabilities = _sigmoid(np.asarray(scores, dtype=np.float64))
    predictions = np.asarray(scores) >= threshold
    return [
        {
            "row_id": row_id,
            "label": int(label),
            "score": float(score),
            "probability": float(probability),
            "prediction": int(prediction),
        }
        for row_id, label, score, probability, prediction in zip(
            row_ids, labels, scores, probabilities, predictions, strict=True
        )
    ]


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.digitize(probabilities, boundaries[1:-1])
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            error += np.mean(mask) * abs(np.mean(labels[mask]) - np.mean(probabilities[mask]))
    return float(error)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _percentage_name(value: float) -> str:
    return f"{value * 100:g}pct"
