from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from probe_transfer.alignment.task_adaptation import is_repeated_control


def alignment_recovery_record(
    context: dict[str, Any],
    labels: np.ndarray,
    oracle: np.ndarray,
    raw: np.ndarray,
    aligned: np.ndarray,
    source_oracle_auroc: float,
    config: dict[str, Any],
    bootstrap_seed: int,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    oracle_auroc = float(roc_auc_score(labels, oracle))
    raw_auroc = float(roc_auc_score(labels, raw))
    aligned_auroc = float(roc_auc_score(labels, aligned))
    raw_gap = oracle_auroc - raw_auroc
    improvement = aligned_auroc - raw_auroc
    residual = oracle_auroc - aligned_auroc
    recovery = improvement / raw_gap if raw_gap > 0 else None
    samples = evaluation["bootstrap_samples"]
    if is_repeated_control(str(context.get("method", ""))):
        samples = evaluation.get("control_bootstrap_samples", samples)
    intervals = _bootstrap_intervals(
        labels,
        oracle,
        raw,
        aligned,
        samples=samples,
        confidence=evaluation["confidence_level"],
        seed=bootstrap_seed,
    )
    substantial = bool(
        source_oracle_auroc >= evaluation["oracle_gate"]
        and oracle_auroc >= evaluation["oracle_gate"]
        and raw_gap >= evaluation["minimum_raw_gap"]
        and improvement >= evaluation["minimum_improvement"]
        and intervals["improvement_ci_lower"] > 0
        and recovery is not None
        and recovery >= evaluation["minimum_recovery"]
    )
    return {
        **context,
        "source_oracle_auroc": source_oracle_auroc,
        "target_oracle_auroc": oracle_auroc,
        "raw_transfer_auroc": raw_auroc,
        "aligned_auroc": aligned_auroc,
        "raw_auroc_gap": raw_gap,
        "aligned_auroc_improvement": improvement,
        "residual_auroc_gap": residual,
        "recovery_fraction": recovery,
        **intervals,
        "substantial_recovery": substantial,
    }


def _bootstrap_intervals(
    labels: np.ndarray,
    oracle: np.ndarray,
    raw: np.ndarray,
    aligned: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    score_sets = [np.asarray(values, dtype=np.float64) for values in (oracle, raw, aligned)]
    if any(len(values) != len(labels) for values in score_sets):
        raise ValueError("Every score array must align with the labels.")
    rng = np.random.default_rng(seed)
    improvements: list[float] = []
    residuals: list[float] = []
    recoveries: list[float] = []
    while len(improvements) < samples:
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        oracle_auc, raw_auc, aligned_auc = (
            roc_auc_score(sampled_labels, values[indices]) for values in score_sets
        )
        gap = oracle_auc - raw_auc
        improvements.append(aligned_auc - raw_auc)
        residuals.append(oracle_auc - aligned_auc)
        recoveries.append((aligned_auc - raw_auc) / gap if gap > 1e-12 else np.nan)
    lower = (1 - confidence) / 2
    upper = 1 - lower
    return {
        "improvement_ci_lower": float(np.quantile(improvements, lower)),
        "improvement_ci_upper": float(np.quantile(improvements, upper)),
        "residual_ci_lower": float(np.quantile(residuals, lower)),
        "residual_ci_upper": float(np.quantile(residuals, upper)),
        "recovery_ci_lower": float(np.nanquantile(recoveries, lower)),
        "recovery_ci_upper": float(np.nanquantile(recoveries, upper)),
    }
