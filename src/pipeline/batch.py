import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core.config import ConfigError
from core.constants import BASELINE_ARTIFACT_ENV, EXPERIMENT_OUTPUT_ENV, PROJECT_ROOT
from pipeline.config import materialize_stage
from pipeline.materials import prepare_panel_materials
from pipeline.panel import select_task
from probe_transfer.alignment.contrasts import condition_contrasts, validate_contrasts
from probe_transfer.alignment.control_decomposition import summarize_control_decomposition
from probe_transfer.artifacts import write_jsonl
from probe_transfer.layout import study_prefix
from probe_transfer.publication import Publication, publish_artifacts


def run_alignment_panel(study: dict[str, Any], path: Path) -> None:
    validate_contrasts(study)
    if not study.get("reuse_materials") or not study.get("decision_rules", {}).get(
        "heldout_requires_included_compatibility"
    ):
        raise ConfigError("Alignment batches require reused materials and a compatibility gate.")
    configured_root = os.getenv(EXPERIMENT_OUTPUT_ENV)
    if not configured_root:
        raise RuntimeError(f"{EXPERIMENT_OUTPUT_ENV} is required for panel output.")
    root = Path(configured_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    conditions = [name for name, value in study["fit_conditions"].items() if value is not None]
    fit_tasks = [name for name, spec in study["tasks"].items() if spec and spec["role"] == "fit"]
    held_out = [
        name for name, spec in study["tasks"].items() if spec and spec["role"] == "held_out"
    ]
    if len(fit_tasks) < 2 or not conditions:
        raise ConfigError(
            "A compatibility batch requires at least two fitting tasks and conditions."
        )
    for task in (*fit_tasks, *held_out):
        for condition in conditions:
            materialize_stage(select_task(study, task, condition), "align")
    workers = study.get("execution", {}).get("alignment_workers", 1)
    if type(workers) is not int or not 1 <= workers <= len(fit_tasks):
        raise ConfigError("Alignment workers must be bounded by the number of fitting tasks.")
    if study.get("execution", {}).get("prepare_materials", False):
        prepare_panel_materials(study, path, root, fit_tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        included = list(
            executor.map(lambda task: _run_task(study, path, root, task, conditions), fit_tasks)
        )
        comparisons = [row for rows in included for row in rows]
        _assert_shared_maps(comparisons)
        eligible = eligible_conditions(comparisons, fit_tasks, conditions)
        _emit("compatibility_checked", eligible_conditions=eligible)
        excluded = [condition for condition in conditions if condition not in eligible]
        if eligible and study.get("execution", {}).get("prepare_materials", False):
            prepare_panel_materials(study, path, root, held_out)
        heldout_results = (
            list(executor.map(lambda task: _run_task(study, path, root, task, eligible), held_out))
            if eligible
            else []
        )
        comparisons.extend(row for rows in heldout_results for row in rows)
    _assert_shared_maps(comparisons)
    summary = {
        "study": study["name"],
        "interpretation": study["decision_rules"]["interpretation"],
        "primary_condition": study["decision_rules"]["primary_condition"],
        "eligible_conditions": eligible,
        "heldout_skipped_conditions": excluded,
        "comparisons": comparisons,
    }
    results = root / "results"
    results.mkdir(exist_ok=True)
    contrasts = condition_contrasts(root, study, fit_tasks)
    if contrasts:
        write_jsonl(results / "contrasts.jsonl", contrasts)
        summary["contrast_rows"] = len(contrasts)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    config = materialize_stage(select_task(study, fit_tasks[0], conditions[0]), "align")
    prefix = study_prefix(config["name"], config["study"])
    publish_artifacts(config, [Publication(results, f"{prefix}/results")], None)
    _emit("panel_complete", comparisons=len(comparisons), heldout_conditions=eligible)


def _run_task(study, path, root, task, conditions) -> list[dict[str, Any]]:
    outcomes = []
    materials = root / "materials" / task
    materials.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        output = root / task / condition
        output.mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ,
            BASELINE_ARTIFACT_ENV: str(materials),
            EXPERIMENT_OUTPUT_ENV: str(output),
        }
        command = [
            sys.executable,
            str(PROJECT_ROOT / "src/run.py"),
            str(path.resolve()),
            "align",
            "--task",
            task,
            "--fit",
            condition,
        ]
        if (output / "results").is_dir():
            command.append("--publish-only")
        _emit("stage_started", task=task, condition=condition)
        with (output / "worker.log").open("a") as log:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        config = materialize_stage(select_task(study, task, condition), "align")
        controls = config["alignment"].get("task_adaptation", {}).get("controls")
        rules = study["decision_rules"][
            "pairing_specificity"
            if controls is not None
            else (
                "pooled_compatibility"
                if study["tasks"][task]["role"] == "fit"
                else "pooled_generalization"
            )
        ]
        result = summarize_alignment(output / "results", config, rules)
        outcomes.append({"task": task, "condition": condition, **result})
        _emit(
            "stage_published",
            task=task,
            condition=condition,
            median_recovery=result["median_recovery"],
            passes_criterion=result["passes_criterion"],
        )
    return outcomes


def summarize_alignment(
    results: Path, config: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    alignment = config["alignment"]
    rows = _read_rows(results / "recovery.jsonl")
    primary = [
        row
        for row in rows
        if row["depth"] == alignment["primary_depth"]
        and row["probe_family"] == alignment["primary_probe_family"]
        and row["pair_group"] == config["evaluation"]["primary_pair_group"]
    ]
    controls = alignment.get("task_adaptation", {}).get("controls")
    if controls is not None:
        result = summarize_control_decomposition(primary, config, rules)
    else:
        result = _legacy_summary(primary, config, rules)
    selected = [row for row in _read_rows(results / "alignment_selection.jsonl") if row["selected"]]
    signatures = {}
    for row in selected:
        key = f"{row['data_seed']}/{row['source_model']}/{row['target_model']}/{row['depth']}/{row['method']}"
        if key in signatures and signatures[key] != row["map_fingerprint"]:
            raise ValueError("A selected map differs across fitting-task diagnostics.")
        signatures[key] = row["map_fingerprint"]
    if len(signatures) != len(config["data_seeds"]) * len(config["alignment"]["depths"]) * sum(
        len(pairs) for pairs in config["evaluation"]["pair_groups"].values()
    ) * (len([name for name in config["alignment"]["methods"] if name != "quotient_ridge"]) + 1):
        raise ValueError("Selected map identity records are incomplete.")
    return {**result, "map_fingerprints": signatures}


def _legacy_summary(primary, config, rules) -> dict[str, Any]:
    alignment = config["alignment"]
    aligned = [row for row in primary if row["method"] == alignment["primary_method"]]
    shuffled = [row for row in primary if row["method"] == alignment["negative_control"]]
    expected = {
        (seed, source, target)
        for seed in config["data_seeds"]
        for source, target in config["evaluation"]["pair_groups"][
            config["evaluation"]["primary_pair_group"]
        ]
    }
    for values in (aligned, shuffled):
        actual = {(row["data_seed"], row["source_model"], row["target_model"]) for row in values}
        if len(values) != len(expected) or actual != expected:
            raise ValueError("Primary comparisons are missing or duplicated.")
    result = {
        "median_recovery": _median(aligned, "recovery_fraction"),
        "median_retention": _median(aligned, "improvement_retention"),
        "median_aligned_auroc": _median(aligned, "aligned_auroc"),
        "substantial": sum(row["substantial_recovery"] for row in aligned),
        "shuffled_substantial": sum(row["substantial_recovery"] for row in shuffled),
    }
    result["passes_criterion"] = bool(
        result["median_recovery"] is not None
        and result["median_retention"] is not None
        and result["median_recovery"] >= rules["minimum_median_recovery"]
        and result["median_retention"] >= rules["minimum_median_retention"]
        and result["substantial"] >= rules["minimum_substantial"]
        and result["shuffled_substantial"] <= rules["maximum_shuffled_substantial"]
    )
    return result


def eligible_conditions(comparisons, fit_tasks, conditions) -> list[str]:
    return [
        condition
        for condition in conditions
        if all(
            len(
                matches := [
                    row
                    for row in comparisons
                    if row["task"] == task and row["condition"] == condition
                ]
            )
            == 1
            and matches[0]["passes_criterion"]
            for task in fit_tasks
        )
    ]


def _assert_shared_maps(comparisons) -> None:
    signatures = {}
    for row in comparisons:
        condition = row["condition"]
        if condition in signatures and signatures[condition] != row["map_fingerprints"]:
            raise ValueError("A shared map changed with the evaluation task.")
        signatures[condition] = row["map_fingerprints"]


def _median(rows, field) -> float | None:
    values = [row[field] for row in rows]
    return None if any(value is None for value in values) else statistics.median(values)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _emit(event: str, **details: Any) -> None:
    print(json.dumps({"event": event, **details}, allow_nan=False), flush=True)
