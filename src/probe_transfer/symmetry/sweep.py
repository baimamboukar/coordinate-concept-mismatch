from collections import defaultdict
from itertools import pairwise
from math import sqrt
from typing import Any


def scale_sweep_metrics(
    recoveries: list[dict[str, Any]], symmetry: dict[str, Any]
) -> dict[str, float]:
    settings = symmetry.get("dose_response")
    if settings is None:
        return {}

    variants = settings["ordered_variants"]
    by_variant = {variant: [] for variant in variants}
    trajectories: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in recoveries:
        variant = row["transformation_variant"]
        by_variant[variant].append(row)
        key = (
            row["data_seed"],
            row["model"],
            row["depth"],
            row["probe_family"],
            row["transformation_seed"],
        )
        trajectories[key][variant] = float(row["raw_auroc_gap"])

    means = [
        sum(float(row["raw_auroc_gap"]) for row in by_variant[v]) / len(by_variant[v])
        for v in variants
    ]
    minimum_rho = settings["minimum_trajectory_spearman"]
    monotonic = 0
    for values in trajectories.values():
        if set(values) != set(variants):
            raise ValueError("Every scale trajectory must contain every configured range.")
        statistic = _spearman([values[v] for v in variants])
        monotonic += statistic >= minimum_rho

    failure_counts = [
        sum(bool(row["coordinate_failure"]) for row in by_variant[v]) for v in variants
    ]
    first_crossing = next(
        (
            index
            for index, count in enumerate(failure_counts)
            if count >= settings["minimum_failure_comparisons"]
        ),
        -1,
    )
    mean_non_decreasing = all(after >= before for before, after in pairwise(means))
    metrics = {
        "scale_sweep/mean_non_decreasing": float(mean_non_decreasing),
        "scale_sweep/monotonic_trajectory_fraction": monotonic / len(trajectories),
        "scale_sweep/dose_response_supported": float(
            mean_non_decreasing and monotonic >= settings["minimum_monotonic_trajectories"]
        ),
        "scale_sweep/first_robust_failure_index": float(first_crossing),
    }
    for variant, rows, mean, failures in zip(
        variants, (by_variant[v] for v in variants), means, failure_counts, strict=True
    ):
        metrics.update(
            {
                f"scale_sweep/{variant}/mean_raw_auroc_gap": mean,
                f"scale_sweep/{variant}/coordinate_failure_fraction": failures / len(rows),
                f"scale_sweep/{variant}/exact_recovery_fraction": sum(
                    bool(row["exact_recovery"]) for row in rows
                )
                / len(rows),
                f"scale_sweep/{variant}/estimated_recovery_fraction": sum(
                    bool(row.get("estimated_recovery")) for row in rows
                )
                / len(rows),
            }
        )
    return metrics


def _spearman(values: list[float]) -> float:
    ranks = _average_ranks(values)
    positions = [float(index) for index in range(len(values))]
    position_mean = sum(positions) / len(positions)
    rank_mean = sum(ranks) / len(ranks)
    numerator = sum(
        (position - position_mean) * (rank - rank_mean)
        for position, rank in zip(positions, ranks, strict=True)
    )
    position_norm = sum((value - position_mean) ** 2 for value in positions)
    rank_norm = sum((value - rank_mean) ** 2 for value in ranks)
    denominator = sqrt(position_norm * rank_norm)
    return numerator / denominator if denominator else 0.0


def _average_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        rank = (start + stop - 1) / 2
        for index, _ in ordered[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks
