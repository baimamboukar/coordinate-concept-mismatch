import numpy as np

from probe_transfer.probes.evaluation import (
    binary_metrics,
    fixed_operating_point_metrics,
    paired_auroc_gap_interval,
    prediction_rows,
    select_balanced_accuracy_threshold,
    select_fpr_thresholds,
)


def test_binary_metrics_retain_confusion_and_low_fpr_evidence() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([-2.0, -1.0, 1.0, 2.0])
    threshold = select_balanced_accuracy_threshold(labels, scores)

    metrics = binary_metrics(labels, scores, threshold=threshold)

    assert metrics["auroc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 0, 2)
    assert metrics["tpr_at_fpr_1pct"] == 1.0


def test_source_fpr_thresholds_can_be_frozen_on_target() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([-2.0, -1.0, 1.0, 2.0])

    thresholds = select_fpr_thresholds(labels, scores, [0.01, 0.05])
    metrics = fixed_operating_point_metrics(labels, scores, thresholds)

    assert metrics["achieved_fpr_at_source_1pct"] == 0.0
    assert metrics["tpr_at_source_1pct"] == 1.0


def test_source_fpr_thresholds_are_finite_for_uninformative_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.zeros(4)

    thresholds = select_fpr_thresholds(labels, scores, [0.01])

    assert np.isfinite(thresholds[0.01])


def test_prediction_rows_include_recomputable_fields() -> None:
    rows = prediction_rows(["a", "b"], np.array([0, 1]), np.array([-1.0, 1.0]), 0.0)

    assert set(rows[0]) == {"row_id", "label", "score", "probability", "prediction"}
    assert [row["prediction"] for row in rows] == [0, 1]


def test_paired_bootstrap_interval_preserves_score_pairing() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    oracle = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    transfer = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])

    gap, lower, upper = paired_auroc_gap_interval(
        labels, oracle, transfer, samples=100, confidence=0.95, seed=42
    )

    assert (gap, lower, upper) == (1.0, 1.0, 1.0)
