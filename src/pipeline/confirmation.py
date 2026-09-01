from typing import Any


def endpoints_from_summary(
    model_pair: str, summary: dict[str, Any], tasks: list[str]
) -> list[dict[str, Any]]:
    qualifications = summary.get("qualifications", {})
    comparisons = {row["task"]: row for row in summary.get("comparisons", [])}
    endpoints = []
    for task in tasks:
        qualification = qualifications.get(task)
        if not isinstance(qualification, dict) or type(qualification.get("qualified")) is not bool:
            raise ValueError(f"Missing qualification result for {model_pair}/{task}.")
        comparison = comparisons.get(task, {})
        endpoints.append(
            {
                "model_pair": model_pair,
                "task": task,
                "qualified": qualification["qualified"],
                "passes_criterion": comparison.get("passes_criterion", False),
                "pooled_empirical_p": comparison.get("pooled_empirical_p"),
            }
        )
    return endpoints


def synthesize_independent_confirmation(
    endpoints: list[dict[str, Any]], rules: dict[str, Any]
) -> dict[str, Any]:
    total = rules["total_endpoints"]
    if len(endpoints) != total:
        raise ValueError(f"Expected {total} independent-confirmation endpoints.")
    keys = [(row["model_pair"], row["task"]) for row in endpoints]
    if len(set(keys)) != total:
        raise ValueError("Independent-confirmation endpoints must be unique.")

    probabilities = []
    for row in endpoints:
        if type(row.get("qualified")) is not bool or type(row.get("passes_criterion")) is not bool:
            raise ValueError("Endpoint decisions must be boolean.")
        probability = row.get("pooled_empirical_p")
        if not row["qualified"]:
            probability = 1.0
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            raise TypeError("Qualified endpoints require a numeric pooled probability.")
        value = float(probability)
        if not 0 <= value <= 1:
            raise ValueError("Qualified endpoints require a valid pooled probability.")
        probabilities.append(value)

    adjusted = holm_adjust(probabilities)
    alpha = float(rules["familywise_alpha"])
    resolved = []
    for row, probability, adjusted_probability in zip(
        endpoints, probabilities, adjusted, strict=True
    ):
        passes = bool(
            row["qualified"] and row["passes_criterion"] and adjusted_probability <= alpha
        )
        resolved.append(
            {
                **row,
                "pooled_empirical_p": probability,
                "holm_adjusted_p": adjusted_probability,
                "passes_endpoint": passes,
            }
        )

    passed = [row for row in resolved if row["passes_endpoint"]]
    task_coverage = all(
        any(row["task"] == task for row in passed) for task in {r["task"] for r in resolved}
    )
    pair_coverage = all(
        any(row["model_pair"] == pair for row in passed)
        for pair in {r["model_pair"] for r in resolved}
    )
    confirmed = bool(
        len(passed) >= rules["minimum_endpoint_passes"]
        and (task_coverage or not rules["require_each_task"])
        and (pair_coverage or not rules["require_each_model_pair"])
    )
    return {
        "confirmed": confirmed,
        "endpoint_passes": len(passed),
        "task_coverage": task_coverage,
        "model_pair_coverage": pair_coverage,
        "endpoints": resolved,
    }


def holm_adjust(probabilities: list[float]) -> list[float]:
    order = sorted(range(len(probabilities)), key=probabilities.__getitem__)
    adjusted = [0.0] * len(probabilities)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(probabilities) - rank) * probabilities[index]))
        adjusted[index] = running
    return adjusted
