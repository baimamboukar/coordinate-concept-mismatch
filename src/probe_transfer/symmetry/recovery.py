from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from probe_transfer.probes.evaluation import paired_auroc_gap_interval


def recovery_record(
    context: dict[str, Any],
    permutation_seed: int,
    labels: np.ndarray,
    reference: np.ndarray,
    raw: np.ndarray,
    transported: np.ndarray,
    inverse: np.ndarray,
    estimated: np.ndarray | None,
    config: dict[str, Any],
    comparison_index: int,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    gap, lower, upper = paired_auroc_gap_interval(
        labels,
        reference,
        raw,
        samples=evaluation["bootstrap_samples"],
        confidence=evaluation["confidence_level"],
        seed=config["seed"] + comparison_index,
    )
    reference_auroc = float(roc_auc_score(labels, reference))
    raw_auroc = float(roc_auc_score(labels, raw))
    transported_auroc = float(roc_auc_score(labels, transported))
    inverse_auroc = float(roc_auc_score(labels, inverse))
    recovery = (transported_auroc - raw_auroc) / gap if gap > 0 else None
    score_error = float(np.max(np.abs(transported - reference)))
    score_match = bool(
        np.allclose(
            transported,
            reference,
            atol=evaluation["score_atol"],
            rtol=evaluation["score_rtol"],
        )
    )
    coordinate_failure = bool(
        reference_auroc >= evaluation["oracle_gate"]
        and gap >= evaluation["minimum_gap"]
        and lower > 0
    )
    exact_recovery = bool(
        gap > 0
        and score_match
        and abs(transported_auroc - reference_auroc) <= evaluation["auroc_recovery_atol"]
        and recovery is not None
        and recovery >= evaluation["minimum_recovery"]
    )
    record = {
        **context,
        "permutation_seed": permutation_seed,
        "reference_auroc": reference_auroc,
        "raw_auroc": raw_auroc,
        "transported_auroc": transported_auroc,
        "inverse_auroc": inverse_auroc,
        "raw_auroc_gap": gap,
        "ci_lower": lower,
        "ci_upper": upper,
        "recovery_fraction": recovery,
        "maximum_score_error": score_error,
        "score_match": score_match,
        "coordinate_failure": coordinate_failure,
        "exact_recovery": exact_recovery,
    }
    if estimated is not None:
        estimated_auroc = float(roc_auc_score(labels, estimated))
        estimated_fraction = (estimated_auroc - raw_auroc) / gap if gap > 0 else None
        estimated_score_match = bool(
            np.allclose(
                estimated,
                reference,
                atol=evaluation["score_atol"],
                rtol=evaluation["score_rtol"],
            )
        )
        record.update(
            {
                "estimated_auroc": estimated_auroc,
                "estimated_recovery_fraction": estimated_fraction,
                "estimated_maximum_score_error": float(np.max(np.abs(estimated - reference))),
                "estimated_score_match": estimated_score_match,
                "estimated_recovery": bool(
                    gap > 0
                    and estimated_fraction is not None
                    and estimated_fraction >= evaluation["minimum_recovery"]
                    and abs(estimated_auroc - reference_auroc) <= evaluation["auroc_recovery_atol"]
                ),
            }
        )
    return record
