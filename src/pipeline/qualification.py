import json
import statistics
from pathlib import Path
from typing import Any

from core.config import ConfigError
from pipeline.config import materialize_stage
from pipeline.panel import select_task


def qualify_tasks(root: Path, study: dict[str, Any], tasks: list[str]) -> dict[str, dict[str, Any]]:
    rules = qualification_rules(study)
    return {task: qualify_task(root, study, task, rules) for task in tasks}


def qualify_task(
    root: Path,
    study: dict[str, Any],
    task: str,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = rules or qualification_rules(study)
    config = materialize_stage(select_task(study, task), "align")
    alignment = config["alignment"]
    evaluation = config["evaluation"]
    expected = {
        (seed, source, target)
        for seed in config["data_seeds"]
        for source, target in evaluation["pair_groups"][evaluation["primary_pair_group"]]
    }
    common = {
        "depth": alignment["primary_depth"],
        "probe_family": alignment["primary_probe_family"],
        "pair_group": evaluation["primary_pair_group"],
    }
    gaps = _primary(_read(root / "materials" / task / "results" / "transfer_gaps.jsonl"), common)
    recoveries = _primary(_read(root / task / "same_task" / "results" / "recovery.jsonl"), common)
    affine = [row for row in recoveries if row["method"] == alignment["primary_method"]]
    shuffled = [row for row in recoveries if row["method"] == alignment["negative_control"]]
    _complete(gaps, expected, "transfer-gap")
    _complete(affine, expected, "same-task affine")
    _complete(shuffled, expected, "same-task shuffled")

    result = {
        "frozen_failures": sum(bool(row["transfer_failed"]) for row in gaps),
        "median_same_task_recovery": _median([row["recovery_fraction"] for row in affine]),
        "same_task_substantial": sum(bool(row["substantial_recovery"]) for row in affine),
        "shuffled_substantial": sum(bool(row["substantial_recovery"]) for row in shuffled),
    }
    result["qualified"] = bool(
        result["frozen_failures"] >= rules["required_frozen_failures"]
        and result["median_same_task_recovery"] is not None
        and result["median_same_task_recovery"] >= rules["minimum_same_task_median_recovery"]
        and result["same_task_substantial"] >= rules["minimum_same_task_substantial"]
        and result["shuffled_substantial"] <= rules["maximum_shuffled_substantial"]
    )
    return result


def qualification_rules(study: dict[str, Any]) -> dict[str, Any]:
    rules = study.get("decision_rules", {}).get("qualification")
    required = {
        "required_frozen_failures",
        "minimum_same_task_median_recovery",
        "minimum_same_task_substantial",
        "maximum_shuffled_substantial",
    }
    if not isinstance(rules, dict) or set(rules) != required:
        raise ConfigError("Fresh-task adaptation requires the complete qualification rule.")
    counts = (
        rules["required_frozen_failures"],
        rules["minimum_same_task_substantial"],
        rules["maximum_shuffled_substantial"],
    )
    recovery = rules["minimum_same_task_median_recovery"]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ConfigError("Qualification counts must be non-negative integers.")
    if type(recovery) not in {int, float} or not 0 <= recovery <= 1:
        raise ConfigError("Qualification recovery must lie between zero and one.")
    return rules


def _primary(rows: list[dict[str, Any]], expected: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in rows if all(row.get(key) == value for key, value in expected.items())]


def _complete(rows: list[dict[str, Any]], expected: set[tuple[Any, ...]], label: str) -> None:
    actual = {(row["data_seed"], row["source_model"], row["target_model"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(f"{label.capitalize()} qualification rows are incomplete.")


def _median(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return statistics.median(float(value) for value in values if value is not None)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
