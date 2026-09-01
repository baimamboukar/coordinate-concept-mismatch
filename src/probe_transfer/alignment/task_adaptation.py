from math import isfinite
from typing import Any

import numpy as np
import torch

from core.config import ConfigError
from probe_transfer.alignment.methods import AlignmentMap


def adaptation_method(rank: int, rows: int, *, shuffled: bool = False) -> str:
    prefix = "shuffled_low_rank" if shuffled else "low_rank"
    return f"{prefix}_r{rank}_n{rows}"


def adaptation_method_count(alignment: dict[str, Any]) -> int:
    settings = alignment.get("task_adaptation")
    if settings is None:
        return 0
    return 2 * len(settings["ranks"]) * len(settings["calibration_rows"])


def recovery_reference_method(config: dict[str, Any], method: str) -> str | None:
    settings = config["alignment"].get("task_adaptation")
    if settings is not None and method.startswith(("low_rank_", "shuffled_low_rank_")):
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
    if not isinstance(settings, dict) or set(settings) != required:
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
    expected = adaptation_method(settings["confirmatory_rank"], settings["confirmatory_rows"])
    control = adaptation_method(
        settings["confirmatory_rank"], settings["confirmatory_rows"], shuffled=True
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

    residual = source_values - (target_values @ base.weight + base.bias)
    fitted: dict[str, AlignmentMap] = {}
    for rows in settings["calibration_rows"]:
        inputs = target_values[:rows]
        expected = residual[:rows]
        for shuffled in (False, True):
            outputs = expected
            if shuffled:
                generator = torch.Generator().manual_seed(shuffle_seed + rows)
                order = torch.randperm(rows, generator=generator).to(device)
                outputs = expected.index_select(0, order)
            correction, penalty = _ridge_without_intercept(
                inputs, outputs, float(settings["relative_alpha"])
            )
            left, singular, right = torch.linalg.svd(correction, full_matrices=False)
            for rank in settings["ranks"]:
                update = (left[:, :rank] * singular[:rank]) @ right[:rank]
                name = adaptation_method(rank, rows, shuffled=shuffled)
                fitted[name] = AlignmentMap(
                    name,
                    weight=base.weight + update,
                    bias=base.bias,
                    metadata={
                        "calibration_rows": rows,
                        "correction_rank": rank,
                        "ridge_penalty": penalty,
                        "shuffled_pairing": int(shuffled),
                        "correction_frobenius_norm": float(torch.linalg.vector_norm(update).item()),
                    },
                )
    return fitted


def method_metadata(method: str) -> dict[str, Any]:
    shuffled = method.startswith("shuffled_low_rank_")
    prefix = "shuffled_low_rank_r" if shuffled else "low_rank_r"
    if not method.startswith(prefix):
        return {}
    rank, rows = method.removeprefix(prefix).split("_n", 1)
    return {
        "map_class": "shuffled_control" if shuffled else "task_specific_low_rank",
        "correction_rank": int(rank),
        "calibration_rows": int(rows),
    }


def _ridge_without_intercept(
    inputs: torch.Tensor, outputs: torch.Tensor, relative_alpha: float
) -> tuple[torch.Tensor, float]:
    scale = inputs.square().sum(dim=0).mean().clamp_min(torch.finfo(inputs.dtype).eps)
    penalty = relative_alpha * scale
    if len(inputs) < inputs.shape[1]:
        gram = inputs @ inputs.T
        regularized = gram + penalty * torch.eye(
            len(inputs), dtype=inputs.dtype, device=inputs.device
        )
        weight = inputs.T @ torch.linalg.solve(regularized, outputs)
    else:
        gram = inputs.T @ inputs
        regularized = gram + penalty * torch.eye(
            inputs.shape[1], dtype=inputs.dtype, device=inputs.device
        )
        weight = torch.linalg.solve(regularized, inputs.T @ outputs)
    if not torch.isfinite(weight).all():
        raise ValueError("Low-rank ridge correction produced a non-finite map.")
    return weight, float(penalty.item())


def _sorted_positive_integers(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(type(value) is int and value > 0 for value in values)
        and values == sorted(set(values))
    )
