import json
from pathlib import Path
from typing import Any

from core.config import ConfigError

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
    _require_string(fit, "dataset_key", "fit dataset key")
    _require_string(fit, "source_study", "fit source study")
    if fit["dataset_key"] == config["artifacts"]["dataset_key"]:
        raise ConfigError("Cross-task alignment must fit on a different dataset.")
    if type(fit.get("expected_train_rows")) is not int or fit["expected_train_rows"] < 2:
        raise ConfigError("Cross-task alignment requires at least two fit rows.")
    if not isinstance(reference, dict):
        raise ConfigError("Cross-task alignment requires same-task reference materials.")
    for key in ("source_name", "source_study", "source_variant"):
        _require_string(reference, key, f"reference {key}")


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
