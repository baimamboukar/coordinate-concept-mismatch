from collections.abc import Iterator
from typing import Any

from core.config import NAME_PATTERN, ConfigError, merge_config
from core.reproducibility import is_pinned_hf_revision
from probe_transfer.splits import seeded_split_sizes


def select_task(
    study: dict[str, Any], task: str | None = None, fit: str | None = None
) -> dict[str, Any]:
    tasks = study.get("tasks")
    if tasks is None:
        if task is not None or fit is not None:
            raise ConfigError("Task selectors require a configured task panel.")
        return study
    if isinstance(tasks, dict):
        tasks = {name: spec for name, spec in tasks.items() if spec is not None}
    if not isinstance(tasks, dict) or task not in tasks:
        raise ConfigError(f"Select --task from the configured panel: {list(tasks or {})}")
    if not isinstance(task, str) or not NAME_PATTERN.fullmatch(task):
        raise ConfigError("Panel task names must use semantic snake_case.")
    spec = tasks[task]
    if not isinstance(spec, dict) or spec.get("role") not in {"fit", "held_out"}:
        raise ConfigError("Every panel task requires a fit or held_out role.")
    if set(spec) - {"role", "dataset", "sampling", "artifacts"}:
        raise ConfigError("Task overrides may change only data, sampling, and artifact keys.")
    base = {key: value for key, value in study.items() if key not in {"tasks", "fit_conditions"}}
    configured = merge_config(base, {key: value for key, value in spec.items() if key != "role"})
    if not is_pinned_hf_revision(configured["dataset"].get("revision")):
        raise ConfigError("Panel datasets require exact 40-character commit revisions.")
    configured.update(task=task, task_role=spec["role"], fit_condition=fit or "same_task")
    stages = configured["pipeline"]["stages"]
    reuse = study.get("reuse_materials", {})
    if not isinstance(reuse, dict) or set(reuse) - {"study", "transfer", "align"}:
        raise ConfigError("reuse_materials requires study, transfer, and align names.")
    if reuse and (
        set(reuse) != {"study", "transfer", "align"}
        or any(
            not isinstance(value, str) or not NAME_PATTERN.fullmatch(value)
            for value in reuse.values()
        )
    ):
        raise ConfigError("Reused material names must be complete semantic snake_case names.")
    for stage in ("transfer", "align"):
        stages[stage]["artifact_variant"] = f"{task.replace('_', '-')}-{stage}"
    stages["align"]["materials"] = {
        "source_name": reuse.get(
            "transfer", stages["transfer"].get("name", f"{study['name']}_transfer")
        ),
        "source_study": reuse.get("study", study["name"]),
        "source_variant": stages["transfer"]["artifact_variant"],
        **{
            f"expected_{split}_rows": size
            for split, size in {
                **seeded_split_sizes(configured),
                "test": configured["sampling"]["test_size"],
            }.items()
        },
    }
    if fit is not None:
        _select_fit(study, configured, task, fit)
    return configured


def task_variants(study: dict[str, Any], stage: str) -> Iterator[dict[str, Any]]:
    if "tasks" not in study:
        yield study
        return
    for task in study["tasks"]:
        if study["tasks"][task] is None:
            continue
        yield select_task(study, task)
        if stage == "align":
            for fit, condition in study.get("fit_conditions", {}).items():
                if condition is None:
                    continue
                if not isinstance(condition, dict):
                    raise ConfigError("Fit conditions must be enabled mappings.")
                sources = condition.get("datasets", condition)
                if set(sources) != {task}:
                    yield select_task(study, task, fit)


def _select_fit(study: dict[str, Any], configured: dict[str, Any], task: str, fit: str) -> None:
    conditions = study.get("fit_conditions", {})
    if fit not in conditions or not NAME_PATTERN.fullmatch(fit):
        raise ConfigError(f"Unknown panel fit condition: {fit}")
    condition = conditions[fit]
    if not isinstance(condition, dict):
        raise ConfigError("Fit conditions must be enabled mappings.")
    sources = condition.get("datasets", condition)
    fitting = condition.get("fitting") if "datasets" in condition else None
    if "datasets" in condition and set(condition) - {
        "datasets",
        "fitting",
        "split",
        "primary_method",
        "methods",
    }:
        raise ConfigError("Structured fit conditions contain an unsupported field.")
    if not isinstance(sources, dict) or not sources or set(sources) == {task}:
        raise ConfigError("Cross-task fits require a distinct fit task.")
    entries = []
    alignment = configured["pipeline"]["stages"]["align"]
    split = condition.get("split", alignment["alignment"].get("fit_split", "train"))
    diagnostic = alignment["alignment"].get("diagnostic_split", "validation")
    if split not in seeded_split_sizes(configured):
        raise ConfigError("Panel fitting must select a configured seeded split.")
    alignment["alignment"]["fit_split"] = split
    for source, rows in sources.items():
        source_config = select_task(study, source)
        if source_config["task_role"] != "fit":
            raise ConfigError("Held-out task activations cannot enter a panel fit condition.")
        sizes = seeded_split_sizes(source_config)
        available = sizes[split]
        if type(rows) is not int or not 2 <= rows <= available:
            raise ConfigError("Panel fit rows must not exceed available training rows.")
        entries.append(
            {
                "dataset_key": source_config["artifacts"]["dataset_key"],
                "source_study": study.get("reuse_materials", {}).get("study", study["name"]),
                "probe_source_name": study.get("reuse_materials", {}).get(
                    "transfer", source_config["pipeline"]["stages"]["transfer"]["name"]
                ),
                "probe_source_variant": source_config["pipeline"]["stages"]["transfer"][
                    "artifact_variant"
                ],
                "expected_train_rows": available,
                "expected_validation_rows": sizes[diagnostic],
                "fit_rows": rows,
            }
        )
    reuse = study.get("reuse_materials", {})
    if fitting is not None:
        alignment["alignment"] = merge_config(alignment["alignment"], {"fitting": fitting})
    if "primary_method" in condition:
        alignment["alignment"]["primary_method"] = condition["primary_method"]
    if "methods" in condition:
        alignment["alignment"]["methods"] = condition["methods"]
    alignment["reference_materials"] = {
        "source_name": reuse.get("align", alignment.get("name", f"{study['name']}_align")),
        "source_study": reuse.get("study", study["name"]),
        "source_variant": alignment["artifact_variant"],
    }
    alignment["artifact_variant"] = f"{fit.replace('_', '-')}-fit-{task.replace('_', '-')}-eval"
    alignment["fit_materials"] = {
        "expected_train_rows": sum(sources.values()),
        "task_balanced": len(entries) > 1 and len(set(sources.values())) == 1,
        "evaluation_included": task in sources,
        "datasets": entries,
    }
