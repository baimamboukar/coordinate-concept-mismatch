import re
from pathlib import Path
from typing import Any

import yaml

from core.constants import (
    CONFIG_SUFFIX,
    EXPERIMENT_NAME_PATTERN,
    GENERIC_EXPERIMENT_PATTERN,
    REQUIRED_BINARY_METRICS,
    REQUIRED_ROW_LEVEL_FIELDS,
)
from core.reproducibility import is_pinned_hf_revision

NAME_PATTERN = re.compile(EXPERIMENT_NAME_PATTERN)
GENERIC_NAME_PATTERN = re.compile(GENERIC_EXPERIMENT_PATTERN)


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix != CONFIG_SUFFIX:
        raise ConfigError("Experiment configurations must use the .yaml extension.")
    if not config_path.is_file():
        raise ConfigError(f"Configuration not found: {config_path}")

    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ConfigError("The configuration root must be a mapping.")

    name = data.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ConfigError("The experiment name must be descriptive snake_case.")
    if GENERIC_NAME_PATTERN.fullmatch(name):
        raise ConfigError("Number-only experiment names are not allowed.")
    if config_path.stem != name:
        raise ConfigError("The YAML filename must match its experiment name.")

    runner = data.get("runner")
    if not isinstance(runner, str) or runner.count(":") != 1:
        raise ConfigError("runner must use the form 'module:function'.")
    if not isinstance(data.get("seed"), int):
        raise ConfigError("seed must be an integer.")
    if not isinstance(data.get("deterministic", True), bool):
        raise ConfigError("deterministic must be a boolean.")

    _validate_huggingface_resources(data.get("models", {}), "model")
    if "dataset" in data:
        _validate_huggingface_resources({"dataset": data["dataset"]}, "dataset")
    _validate_evaluation(data.get("evaluation"))

    return data


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


def _validate_evaluation(evaluation: Any) -> None:
    if not isinstance(evaluation, dict):
        raise ConfigError("Every experiment must declare an evaluation mapping.")

    primary = evaluation.get("primary_metrics")
    secondary = evaluation.get("secondary_metrics")
    retained = evaluation.get("retain_row_level")
    if not isinstance(primary, list) or not primary:
        raise ConfigError("Every experiment must declare at least one primary metric.")
    if not isinstance(secondary, list) or not secondary:
        raise ConfigError("Every experiment must declare secondary metrics.")
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
