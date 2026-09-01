import statistics
from typing import Any

from probe_transfer.alignment.task_adaptation import (
    coral_method,
    residual_shuffle_method,
    source_shuffle_method,
)

Context = tuple[int, str, str]


def summarize_control_decomposition(
    rows: list[dict[str, Any]], config: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    alignment = config["alignment"]
    settings = alignment["task_adaptation"]
    controls = settings["controls"]
    rank = settings["confirmatory_rank"]
    count = settings["confirmatory_rows"]
    residual_repeats = controls["residual_shuffle_repeats"]
    source_repeats = controls["source_shuffle_repeats"]
    expected = {
        (seed, source, target)
        for seed in config["data_seeds"]
        for source, target in config["evaluation"]["pair_groups"][
            config["evaluation"]["primary_pair_group"]
        ]
    }
    indexed = {
        (row["data_seed"], row["source_model"], row["target_model"], row["method"]): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("Primary recovery rows contain duplicates.")

    paired_values, coral_values = [], []
    residual_by_repeat: list[list[float | None]] = [[] for _ in range(residual_repeats)]
    source_values: list[float | None] = []
    details = []
    for context in sorted(expected):
        paired = _row(indexed, context, alignment["primary_method"])
        coral = _row(indexed, context, coral_method(rank, count))
        residuals = [
            _row(indexed, context, residual_shuffle_method(rank, count, repeat))
            for repeat in range(residual_repeats)
        ]
        sources = [
            _row(indexed, context, source_shuffle_method(rank, count, repeat))
            for repeat in range(source_repeats)
        ]
        paired_recovery = paired["recovery_fraction"]
        residual_recovery = [row["recovery_fraction"] for row in residuals]
        source_recovery = [row["recovery_fraction"] for row in sources]
        residual_median = _median(residual_recovery)
        lift = (
            None
            if paired_recovery is None or residual_median is None
            else paired_recovery - residual_median
        )
        empirical_p = _empirical_p(paired_recovery, residual_recovery)
        beats_all = (
            None
            if paired_recovery is None or any(value is None for value in residual_recovery)
            else paired_recovery > max(residual_recovery)
        )
        details.append(
            {
                "data_seed": context[0],
                "source_model": context[1],
                "target_model": context[2],
                "paired_recovery": paired_recovery,
                "paired_retention": paired["improvement_retention"],
                "paired_aligned_auroc": paired["aligned_auroc"],
                "residual_shuffle_median_recovery": residual_median,
                "source_shuffle_median_recovery": _median(source_recovery),
                "coral_recovery": coral["recovery_fraction"],
                "pairing_specific_lift": lift,
                "empirical_p": empirical_p,
                "paired_beats_all_residual_shuffles": beats_all,
            }
        )
        paired_values.append(paired_recovery)
        coral_values.append(coral["recovery_fraction"])
        source_values.extend(source_recovery)
        for repeat, value in enumerate(residual_recovery):
            residual_by_repeat[repeat].append(value)

    paired_median = _median(paired_values)
    pooled_residual = [_median(values) for values in residual_by_repeat]
    result = {
        "median_recovery": paired_median,
        "median_retention": _median([row["paired_retention"] for row in details]),
        "median_aligned_auroc": _median([row["paired_aligned_auroc"] for row in details]),
        "median_pairing_specific_lift": _median([row["pairing_specific_lift"] for row in details]),
        "pooled_empirical_p": _empirical_p(paired_median, pooled_residual),
        "control_wins": sum(row["paired_beats_all_residual_shuffles"] is True for row in details),
        "median_residual_shuffle_recovery": _median(
            [value for values in residual_by_repeat for value in values]
        ),
        "median_source_shuffle_recovery": _median(source_values),
        "median_coral_recovery": _median(coral_values),
        "substantial": sum(
            bool(row["substantial_recovery"])
            for row in [
                _row(indexed, context, alignment["primary_method"]) for context in sorted(expected)
            ]
        ),
        "control_decomposition": details,
    }
    result["passes_criterion"] = _passes(result, rules)
    return result


def _row(indexed, context: Context, method: str) -> dict[str, Any]:
    row = indexed.get((*context, method))
    if row is None:
        raise ValueError(f"Missing control-decomposition row for {context} and {method}.")
    return row


def _median(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return statistics.median(float(value) for value in values if value is not None)


def _empirical_p(observed: float | None, null: list[float | None]) -> float | None:
    if observed is None or not null or any(value is None for value in null):
        return None
    values = [float(value) for value in null if value is not None]
    return (1 + sum(value >= observed for value in values)) / (len(values) + 1)


def _passes(result: dict[str, Any], rules: dict[str, Any]) -> bool:
    fields = (
        "median_recovery",
        "median_retention",
        "median_pairing_specific_lift",
        "pooled_empirical_p",
    )
    return bool(
        all(result[field] is not None for field in fields)
        and result["median_recovery"] >= rules["minimum_median_recovery"]
        and result["median_retention"] >= rules["minimum_median_retention"]
        and result["median_pairing_specific_lift"] >= rules["minimum_median_paired_advantage"]
        and result["pooled_empirical_p"] <= rules["maximum_empirical_p"]
        and result["control_wins"] >= rules["minimum_control_wins"]
    )
