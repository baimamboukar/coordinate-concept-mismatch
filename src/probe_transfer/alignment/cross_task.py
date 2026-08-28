import json
from pathlib import Path
from typing import Any

import numpy as np

from core.config import ConfigError
from probe_transfer.alignment.materials import paired_split

RecoveryKey = tuple[int, float, str, str, str, str, str]


def validate_cross_task_alignment(config: dict[str, Any]) -> None:
    fit = config.get("fit_materials")
    reference = config.get("reference_materials")
    if fit is None:
        if reference is not None:
            raise ConfigError("Same-task references require cross-task fit materials.")
        return
    if not isinstance(fit, dict):
        raise ConfigError("Cross-task alignment fit_materials must be a mapping.")
    entries = fit_material_entries(config)
    if not entries or len({entry.get("dataset_key") for entry in entries}) != len(entries):
        raise ConfigError("Cross-task fit datasets must be non-empty and unique.")
    evaluation_key = config["artifacts"]["dataset_key"]
    for entry in entries:
        _require_string(entry, "dataset_key", "fit dataset key")
        _require_string(entry, "source_study", "fit source study")
        if entry["dataset_key"] == evaluation_key:
            raise ConfigError("Cross-task alignment must fit on a different dataset.")
        available = entry.get("expected_train_rows")
        selected = entry.get("fit_rows", available)
        if (
            type(available) is not int
            or type(selected) is not int
            or not 2 <= selected <= available
        ):
            raise ConfigError("Cross-task fit rows must not exceed available training rows.")
    expected = (
        fit.get("expected_train_rows") if "datasets" in fit else entries[0]["expected_train_rows"]
    )
    if type(expected) is not int or expected != sum(
        entry.get("fit_rows", entry["expected_train_rows"]) for entry in entries
    ):
        raise ConfigError("Cross-task total fit rows must equal the configured dataset rows.")
    if fit.get("task_balanced") is True and (
        len(entries) < 2
        or len({entry.get("fit_rows", entry["expected_train_rows"]) for entry in entries}) != 1
    ):
        raise ConfigError("Task-balanced fits require equal rows from at least two datasets.")
    if not isinstance(reference, dict):
        raise ConfigError("Cross-task alignment requires same-task reference materials.")
    for key in ("source_name", "source_study", "source_variant"):
        _require_string(reference, key, f"reference {key}")


def fit_material_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    fit = config["fit_materials"]
    datasets = fit.get("datasets")
    if datasets is None:
        return [fit]
    if not isinstance(datasets, list) or any(not isinstance(item, dict) for item in datasets):
        raise ConfigError("fit_materials.datasets must be a list of mappings.")
    return datasets


def fit_material_root(config: dict[str, Any], root: Path, entry: dict[str, Any]) -> Path:
    return root / entry["dataset_key"] if "datasets" in config["fit_materials"] else root


def fit_expected_rows(config: dict[str, Any]) -> int:
    fit = config.get("fit_materials")
    return config["materials"]["expected_train_rows"] if fit is None else fit["expected_train_rows"]


def load_fit_split(
    root: Path,
    config: dict[str, Any],
    source: str,
    target: str,
    split: str,
    layer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if config.get("fit_materials") is None:
        return paired_split(root, source, target, split, layer)
    parts = []
    for entry in fit_material_entries(config):
        values = paired_split(fit_material_root(config, root, entry), source, target, split, layer)
        available = entry["expected_train_rows"]
        if any(len(item) != available for item in values):
            raise ValueError(f"Expected {available} available fit rows for {entry['dataset_key']}.")
        rows = entry.get("fit_rows", available)
        parts.append(tuple(item[:rows] for item in values))
    combined = tuple(np.concatenate(items) for items in zip(*parts, strict=True))
    if any(len(item) != fit_expected_rows(config) for item in combined):
        raise ValueError("Combined fit rows do not match the configured total.")
    return combined  # type: ignore[return-value]


def load_recovery_reference(path: Path | None) -> dict[RecoveryKey, dict[str, Any]] | None:
    if path is None:
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Same-task recovery reference contains duplicate rows.")
    return indexed


def add_improvement_retention(
    row: dict[str, Any],
    reference: dict[RecoveryKey, dict[str, Any]] | None,
    *,
    tolerance: float,
) -> dict[str, Any]:
    if reference is None:
        return row
    same_task = reference.get(_key(row))
    if same_task is None:
        raise ValueError("Same-task recovery reference is incomplete.")
    if abs(row["raw_auroc_gap"] - same_task["raw_auroc_gap"]) > tolerance:
        raise ValueError("Cross-task and same-task recovery baselines differ.")
    denominator = same_task["aligned_auroc_improvement"]
    retention = (
        None if abs(denominator) <= tolerance else row["aligned_auroc_improvement"] / denominator
    )
    return {
        **row,
        "same_task_aligned_auroc": same_task["aligned_auroc"],
        "same_task_recovery_fraction": same_task["recovery_fraction"],
        "improvement_retention": retention,
    }


def _key(row: dict[str, Any]) -> RecoveryKey:
    return (
        row["data_seed"],
        row["depth"],
        row["probe_family"],
        row["source_model"],
        row["target_model"],
        row["pair_group"],
        row["method"],
    )


def _require_string(values: dict[str, Any], key: str, label: str) -> None:
    if not isinstance(values.get(key), str) or not values[key]:
        raise ConfigError(f"Cross-task alignment requires a {label}.")
