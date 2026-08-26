import copy
import re
from pathlib import Path
from typing import Any

import yaml

from core.constants import (
    CONFIG_SUFFIX,
    GENERIC_EXPERIMENT_PATTERN,
    PIPELINE_STAGES,
    SEMANTIC_NAME_PATTERN,
)
from core.reproducibility import is_pinned_hf_revision

NAME_PATTERN = re.compile(SEMANTIC_NAME_PATTERN)
GENERIC_NAME_PATTERN = re.compile(GENERIC_EXPERIMENT_PATTERN)


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix != CONFIG_SUFFIX:
        raise ConfigError("Study configurations must use the .yaml extension.")
    if not config_path.is_file():
        raise ConfigError(f"Configuration not found: {config_path}")

    data = _compose(config_path, ())
    _validate_study(data, config_path)
    return data


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _compose(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(item.name for item in (*stack, path))
        raise ConfigError(f"Configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise ConfigError(f"Configuration not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError("The configuration root must be a mapping.")

    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise ConfigError("extends must be a path or list of paths.")

    composed: dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / parent).resolve()
        composed = merge_config(composed, _compose(parent_path, (*stack, path)))
    return merge_config(composed, data)


def _validate_study(data: dict[str, Any], path: Path) -> None:
    name = data.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ConfigError("The study name must use semantic snake_case.")
    if GENERIC_NAME_PATTERN.fullmatch(name):
        raise ConfigError("Number-only study names are not allowed.")
    if path.stem != name:
        raise ConfigError("The YAML filename must match its study name.")

    if not isinstance(data.get("seed"), int):
        raise ConfigError("seed must be an integer.")
    seeds = data.get("data_seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ConfigError("data_seeds must be a non-empty list of integers.")
    if len(seeds) != len(set(seeds)):
        raise ConfigError("data_seeds must be unique.")
    if not isinstance(data.get("deterministic", True), bool):
        raise ConfigError("deterministic must be a boolean.")

    _validate_huggingface_resources(data.get("models", {}), "model")
    if "dataset" in data:
        _validate_huggingface_resources({"dataset": data["dataset"]}, "dataset")
    pipeline = data.get("pipeline")
    if not isinstance(pipeline, dict) or not isinstance(pipeline.get("stages"), dict):
        raise ConfigError("pipeline.stages must be a mapping.")
    if not pipeline["stages"]:
        raise ConfigError("At least one pipeline stage must be configured.")
    unknown = set(pipeline["stages"]) - set(PIPELINE_STAGES)
    if unknown:
        raise ConfigError(f"Unknown pipeline stages: {sorted(unknown)}")
    if any(not isinstance(spec, dict) for spec in pipeline["stages"].values()):
        raise ConfigError("Every pipeline stage must be a mapping.")


def _validate_huggingface_resources(resources: Any, kind: str) -> None:
    if not isinstance(resources, dict):
        raise ConfigError(f"{kind} resources must be a mapping.")

    for name, resource in resources.items():
        if not isinstance(resource, dict):
            raise ConfigError(f"{kind} '{name}' must be a mapping.")
        if resource.get("backend", "huggingface") != "huggingface":
            continue
        if not isinstance(resource.get("id"), str):
            raise ConfigError(f"Hugging Face {kind} '{name}' requires an id.")
        revision = resource.get("revision")
        if not is_pinned_hf_revision(revision):
            raise ConfigError(
                f"Hugging Face {kind} '{name}' requires an exact 40-character commit revision."
            )


def validate_evaluation(evaluation: Any) -> None:
    from core.constants import REQUIRED_BINARY_METRICS, REQUIRED_ROW_LEVEL_FIELDS

    if not isinstance(evaluation, dict):
        raise ConfigError("Every evaluated stage must declare an evaluation mapping.")

    primary = evaluation.get("primary_metrics")
    secondary = evaluation.get("secondary_metrics")
    retained = evaluation.get("retain_row_level")
    if not isinstance(primary, list) or not primary:
        raise ConfigError("Every evaluated stage must declare at least one primary metric.")
    if not isinstance(secondary, list) or not secondary:
        raise ConfigError("Every evaluated stage must declare secondary metrics.")
    if set(primary) & set(secondary):
        raise ConfigError("Primary and secondary metric lists must not overlap.")

    missing_metrics = REQUIRED_BINARY_METRICS - set(secondary)
    if missing_metrics:
        raise ConfigError(f"Missing required binary metrics: {sorted(missing_metrics)}")
    if not isinstance(retained, list):
        raise ConfigError("Evaluation must declare retained row-level fields.")
    missing_fields = REQUIRED_ROW_LEVEL_FIELDS - set(retained)
    if missing_fields:
        raise ConfigError(f"Missing required row-level fields: {sorted(missing_fields)}")
    if evaluation.get("retain_thresholds") is not True:
        raise ConfigError("Evaluated stages must retain decision thresholds.")
    operating_fprs = evaluation.get("operating_fprs")
    if not isinstance(operating_fprs, list) or 0.01 not in operating_fprs:
        raise ConfigError("Evaluation must include the prespecified 1% FPR operating point.")
