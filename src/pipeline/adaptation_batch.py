import json
import os
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core.config import ConfigError
from core.constants import EXPERIMENT_OUTPUT_ENV
from pipeline.batch import _assert_shared_maps, _emit, _run_task
from pipeline.config import materialize_stage
from pipeline.panel import select_task
from probe_transfer.alignment.task_adaptation import method_metadata
from probe_transfer.artifacts import write_jsonl
from probe_transfer.layout import study_prefix
from probe_transfer.publication import Publication, publish_artifacts


def task_adaptation_variants(study: dict[str, Any]):
    conditions, tasks = _validate_panel(study)
    for task in tasks:
        for condition in conditions:
            yield select_task(study, task, condition)


def run_task_adaptation_panel(study: dict[str, Any], path: Path) -> None:
    conditions, tasks = _validate_panel(study)
    configured_root = os.getenv(EXPERIMENT_OUTPUT_ENV)
    if not configured_root:
        raise RuntimeError(f"{EXPERIMENT_OUTPUT_ENV} is required for panel output.")
    root = Path(configured_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for variant in task_adaptation_variants(study):
        materialize_stage(variant, "align")

    workers = study["execution"].get("alignment_workers", 1)
    if type(workers) is not int or not 1 <= workers <= len(tasks):
        raise ConfigError("Adaptation workers must be bounded by held-out task count.")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(
            executor.map(lambda task: _run_task(study, path, root, task, conditions), tasks)
        )
    comparisons = [row for task_rows in outcomes for row in task_rows]
    _assert_shared_maps(comparisons)
    decomposition = []
    for comparison in comparisons:
        for row in comparison.pop("control_decomposition", []):
            decomposition.append(
                {"task": comparison["task"], "condition": comparison["condition"], **row}
            )
    curves = [
        row
        for task in tasks
        for condition in conditions
        for row in _curve_rows(root, study, task, condition)
    ]
    settings = materialize_stage(select_task(study, tasks[0], conditions[0]), "align")["alignment"][
        "task_adaptation"
    ]
    summary = {
        "study": study["name"],
        "upstream_study": study["reuse_materials"]["study"],
        "interpretation": study["decision_rules"]["interpretation"],
        "primary_condition": study["decision_rules"]["primary_condition"],
        "confirmatory_method": (
            f"low_rank_r{settings['confirmatory_rank']}_n{settings['confirmatory_rows']}"
        ),
        "comparisons": comparisons,
        "curve_rows": len(curves),
        "control_decomposition_rows": len(decomposition),
    }
    results = root / "results"
    results.mkdir(exist_ok=True)
    write_jsonl(results / "adaptation_curves.jsonl", curves)
    if decomposition:
        write_jsonl(results / "control_decomposition.jsonl", decomposition)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    config = materialize_stage(select_task(study, tasks[0], conditions[0]), "align")
    prefix = study_prefix(config["name"], config["study"])
    publish_artifacts(config, [Publication(results, f"{prefix}/results")], None)
    _emit(
        "panel_complete",
        comparisons=len(comparisons),
        curve_rows=len(curves),
        control_decomposition_rows=len(decomposition),
    )


def _validate_panel(study: dict[str, Any]) -> tuple[list[str], list[str]]:
    execution = study.get("execution", {})
    if execution.get("panel_mode") != "task_adaptation":
        raise ConfigError("Task-adaptation execution requires its explicit panel mode.")
    if execution.get("prepare_materials") is not False or not study.get("reuse_materials"):
        raise ConfigError("Task adaptation must reuse immutable published materials.")
    conditions = [name for name, value in study.get("fit_conditions", {}).items() if value]
    tasks = [
        name
        for name, spec in study.get("tasks", {}).items()
        if spec and spec.get("role") == "held_out"
    ]
    fit_tasks = [
        name for name, spec in study.get("tasks", {}).items() if spec and spec.get("role") == "fit"
    ]
    if len(conditions) < 1 or len(tasks) < 1 or len(fit_tasks) < 2:
        raise ConfigError("Task adaptation requires shared-map fits and held-out tasks.")
    if any(
        materialize_stage(select_task(study, task, condition), "align")["alignment"]
        .get("task_adaptation", {})
        .get("controls")
        for task in tasks
        for condition in conditions
    ):
        _validate_pairing_rules(study.get("decision_rules", {}).get("pairing_specificity"))
    return conditions, tasks


def _validate_pairing_rules(rules: Any) -> None:
    required = {
        "minimum_median_recovery",
        "minimum_median_retention",
        "minimum_median_paired_advantage",
        "maximum_empirical_p",
        "minimum_control_wins",
    }
    if not isinstance(rules, dict) or set(rules) != required:
        raise ConfigError("Pairing specificity requires the complete locked decision rule.")
    proportions = [rules[key] for key in required if key != "minimum_control_wins"]
    if any(type(value) not in {int, float} or not 0 <= value <= 1 for value in proportions):
        raise ConfigError(
            "Pairing-specific thresholds must be numeric values between zero and one."
        )
    if type(rules["minimum_control_wins"]) is not int or rules["minimum_control_wins"] < 1:
        raise ConfigError("Pairing-specific control wins must be a positive integer.")


def _curve_rows(root: Path, study: dict[str, Any], task: str, condition: str):
    config = materialize_stage(select_task(study, task, condition), "align")
    rows = _read_rows(root / task / condition / "results" / "recovery.jsonl")
    primary = [
        row
        for row in rows
        if row["depth"] == config["alignment"]["primary_depth"]
        and row["probe_family"] == config["alignment"]["primary_probe_family"]
        and row["pair_group"] == config["evaluation"]["primary_pair_group"]
    ]
    grouped = defaultdict(list)
    for row in primary:
        grouped[row["method"]].append(row)
    expected = len(config["data_seeds"]) * len(
        config["evaluation"]["pair_groups"][config["evaluation"]["primary_pair_group"]]
    )
    if not grouped or any(len(values) != expected for values in grouped.values()):
        raise ValueError("Adaptation curves contain incomplete primary comparisons.")
    context = {"task": task, "condition": condition}
    curves = [
        _method_summary(context, method, values) for method, values in sorted(grouped.items())
    ]
    base = grouped[config["alignment"]["task_adaptation"]["base_method"]]
    curves.extend(
        [
            {
                **context,
                "method": "frozen_transfer",
                "map_class": "frozen_transfer",
                "comparisons": len(base),
                "median_aligned_auroc": _median(base, "raw_transfer_auroc"),
                "median_recovery": 0.0,
                "median_retention": 0.0,
                "substantial": 0,
            },
            {
                **context,
                "method": "same_task_affine_oracle",
                "map_class": "same_task_affine_oracle",
                "comparisons": len(base),
                "median_aligned_auroc": _median(base, "same_task_aligned_auroc"),
                "median_recovery": _median(base, "same_task_recovery_fraction"),
                "median_retention": 1.0,
                "substantial": None,
            },
        ]
    )
    return curves


def _method_summary(context, method, rows) -> dict[str, Any]:
    metadata = method_metadata(method)
    if not metadata:
        metadata = {
            "map_class": "shuffled_control" if method.startswith("shuffled") else "shared_global"
        }
    return {
        **context,
        "method": method,
        **metadata,
        "comparisons": len(rows),
        "median_aligned_auroc": _median(rows, "aligned_auroc"),
        "median_recovery": _median(rows, "recovery_fraction"),
        "median_retention": _median(rows, "improvement_retention"),
        "substantial": sum(bool(row["substantial_recovery"]) for row in rows),
    }


def _median(rows, field) -> float | None:
    values = [row[field] for row in rows]
    return None if any(value is None for value in values) else statistics.median(values)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
