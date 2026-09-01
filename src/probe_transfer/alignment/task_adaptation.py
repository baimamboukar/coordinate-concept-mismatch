from math import isfinite
from typing import Any

import numpy as np
import torch

from core.config import ConfigError
from probe_transfer.alignment.correction_fitting import (
    RidgeFactor,
    fit_coral,
    repeat_seed,
    shuffle_order,
    validate_control_settings,
)
from probe_transfer.alignment.methods import AlignmentMap


def adaptation_method(rank: int, rows: int, *, shuffled: bool = False) -> str:
    prefix = "shuffled_low_rank" if shuffled else "low_rank"
    return f"{prefix}_r{rank}_n{rows}"


def residual_shuffle_method(rank: int, rows: int, repeat: int) -> str:
    return f"residual_shuffle_low_rank_r{rank}_n{rows}_rep{repeat:02d}"


def source_shuffle_method(rank: int, rows: int, repeat: int) -> str:
    return f"source_shuffle_low_rank_r{rank}_n{rows}_rep{repeat:02d}"


def coral_method(rank: int, rows: int) -> str:
    return f"coral_low_rank_r{rank}_n{rows}"


def adaptation_method_count(alignment: dict[str, Any]) -> int:
    settings = alignment.get("task_adaptation")
    if settings is None:
        return 0
    endpoints = len(settings["ranks"]) * len(settings["calibration_rows"])
    controls = settings.get("controls")
    if controls is None:
        return 2 * endpoints
    return endpoints * (
        2 + controls["residual_shuffle_repeats"] + controls["source_shuffle_repeats"]
    )


def recovery_reference_method(config: dict[str, Any], method: str) -> str | None:
    settings = config["alignment"].get("task_adaptation")
    if settings is not None and method_metadata(method):
        return str(settings["reference_method"])
    return None


def validate_task_adaptation(config: dict[str, Any]) -> None:
    alignment = config["alignment"]
    settings = alignment.get("task_adaptation")
    if settings is None:
        return
    required = {
        "base_method",
        "calibration_split",
        "calibration_rows",
        "ranks",
        "confirmatory_rows",
        "confirmatory_rank",
        "relative_alpha",
        "reference_method",
    }
    if (
        not isinstance(settings, dict)
        or not required <= set(settings)
        or set(settings) - (required | {"controls"})
    ):
        raise ConfigError("Task adaptation requires the complete low-rank correction contract.")
    if (
        settings["base_method"] != "affine_ridge"
        or settings["base_method"] not in alignment["methods"]
    ):
        raise ConfigError("Low-rank correction currently requires a fitted affine-ridge base map.")
    if settings["reference_method"] != "affine_ridge" or not config.get("fit_materials"):
        raise ConfigError("Task adaptation requires the full same-task affine reference.")

    split = settings["calibration_split"]
    available = config.get("materials", {}).get(f"expected_{split}_rows")
    rows = settings["calibration_rows"]
    if (
        not isinstance(split, str)
        or type(available) is not int
        or not _sorted_positive_integers(rows)
        or rows[-1] > available
        or settings["confirmatory_rows"] not in rows
    ):
        raise ConfigError("Adaptation row budgets must be sorted and fit the calibration split.")
    width = next(iter(config["models"].values()))["hidden_size"]
    ranks = settings["ranks"]
    if (
        not _sorted_positive_integers(ranks)
        or ranks[-1] > width
        or settings["confirmatory_rank"] not in ranks
    ):
        raise ConfigError("Adaptation ranks must be sorted, unique, and width-bounded.")
    alpha = settings["relative_alpha"]
    if type(alpha) not in {int, float} or not isfinite(alpha) or alpha <= 0:
        raise ConfigError("Adaptation relative alpha must be finite and positive.")
    if split == alignment.get("diagnostic_split"):
        raise ConfigError("Adaptation fitting and diagnosis require disjoint splits.")

    controls = settings.get("controls")
    expected = adaptation_method(settings["confirmatory_rank"], settings["confirmatory_rows"])
    control = adaptation_method(
        settings["confirmatory_rank"], settings["confirmatory_rows"], shuffled=True
    )
    if controls is not None:
        validate_control_settings(controls, rows)
        samples = config["evaluation"].get("control_bootstrap_samples")
        if (
            type(samples) is not int
            or not 1 <= samples <= config["evaluation"]["bootstrap_samples"]
        ):
            raise ConfigError(
                "Repeated controls require a positive bounded control bootstrap count."
            )
        control = residual_shuffle_method(
            settings["confirmatory_rank"], settings["confirmatory_rows"], 0
        )
    if alignment.get("primary_method") != expected or alignment.get("negative_control") != control:
        raise ConfigError("Primary and negative-control methods must match the locked endpoint.")


def fit_task_adaptations(
    base: AlignmentMap,
    source: np.ndarray,
    target: np.ndarray,
    settings: dict[str, Any],
    *,
    shuffle_seed: int,
    device: str,
) -> dict[str, AlignmentMap]:
    if base.weight is None or base.bias is None:
        raise ValueError("Task adaptation requires a complete affine base map.")
    if (
        source.ndim != 2
        or source.shape != target.shape
        or len(source) < max(settings["calibration_rows"])
    ):
        raise ValueError("Task adaptation requires paired, width-matched calibration arrays.")
    source_values = torch.as_tensor(source, dtype=torch.float32, device=device)
    target_values = torch.as_tensor(target, dtype=torch.float32, device=device)
    if not torch.isfinite(source_values).all() or not torch.isfinite(target_values).all():
        raise ValueError("Task-adaptation activations must be finite.")
    if base.weight.shape != (source.shape[1], source.shape[1]) or base.bias.shape != (
        source.shape[1],
    ):
        raise ValueError("The shared map and task activations have incompatible widths.")

    mapped = target_values @ base.weight + base.bias
    residual = source_values - mapped
    fitted: dict[str, AlignmentMap] = {}
    for rows in settings["calibration_rows"]:
        inputs = target_values[:rows]
        ridge = RidgeFactor.prepare(inputs, float(settings["relative_alpha"]))
        _add_low_rank_maps(
            fitted,
            base,
            ridge,
            residual[:rows],
            settings["ranks"],
            rows,
            adaptation_method,
            {},
        )
        controls = settings.get("controls")
        if controls is None:
            order = shuffle_order(rows, shuffle_seed + rows, inputs.device)
            _add_low_rank_maps(
                fitted,
                base,
                ridge,
                residual[:rows].index_select(0, order),
                settings["ranks"],
                rows,
                lambda rank, count: adaptation_method(rank, count, shuffled=True),
                {"shuffled_pairing": 1},
            )
            continue
        for repeat in range(controls["residual_shuffle_repeats"]):
            order = shuffle_order(rows, repeat_seed(shuffle_seed, rows, repeat), inputs.device)
            _add_low_rank_maps(
                fitted,
                base,
                ridge,
                residual[:rows].index_select(0, order),
                settings["ranks"],
                rows,
                lambda rank, count, rep=repeat: residual_shuffle_method(rank, count, rep),
                {"control_repeat": repeat, "row_correspondence": 0},
            )
        for repeat in range(controls["source_shuffle_repeats"]):
            order = shuffle_order(rows, repeat_seed(shuffle_seed, rows, repeat), inputs.device)
            outputs = source_values[:rows].index_select(0, order) - mapped[:rows]
            _add_low_rank_maps(
                fitted,
                base,
                ridge,
                outputs,
                settings["ranks"],
                rows,
                lambda rank, count, rep=repeat: source_shuffle_method(rank, count, rep),
                {"control_repeat": repeat, "row_correspondence": 0},
            )
        for rank in settings["ranks"]:
            name = coral_method(rank, rows)
            fitted[name] = fit_coral(
                base,
                source_values[:rows],
                target_values[:rows],
                rank,
                rows,
                float(controls["covariance_shrinkage"]),
                name,
            )
    return fitted


def method_metadata(method: str) -> dict[str, Any]:
    patterns = (
        ("residual_shuffle_low_rank_r", "residual_shuffle_control"),
        ("source_shuffle_low_rank_r", "source_shuffle_control"),
        ("shuffled_low_rank_r", "shuffled_control"),
        ("coral_low_rank_r", "unpaired_moment_matching"),
        ("low_rank_r", "task_specific_low_rank"),
    )
    for prefix, map_class in patterns:
        if not method.startswith(prefix):
            continue
        payload = method.removeprefix(prefix)
        repeat = None
        if "_rep" in payload:
            payload, repeat_value = payload.rsplit("_rep", 1)
            repeat = int(repeat_value)
        rank, rows = payload.split("_n", 1)
        metadata: dict[str, Any] = {
            "map_class": map_class,
            "correction_rank": int(rank),
            "calibration_rows": int(rows),
        }
        if repeat is not None:
            metadata["control_repeat"] = repeat
        return metadata
    return {}


def is_repeated_control(method: str) -> bool:
    return method.startswith(("residual_shuffle_low_rank_", "source_shuffle_low_rank_"))


def _add_low_rank_maps(fitted, base, ridge, outputs, ranks, rows, naming, metadata) -> None:
    for rank, update in ridge.updates(outputs, ranks).items():
        name = naming(rank, rows)
        fitted[name] = AlignmentMap(
            name,
            weight=base.weight + update,
            bias=base.bias,
            metadata={
                "calibration_rows": rows,
                "correction_rank": rank,
                "ridge_penalty": ridge.penalty,
                **metadata,
                "correction_frobenius_norm": float(torch.linalg.vector_norm(update).item()),
            },
        )


def _sorted_positive_integers(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(type(value) is int and value > 0 for value in values)
        and values == sorted(set(values))
    )
