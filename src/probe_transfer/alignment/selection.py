from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.config import ConfigError
from probe_transfer.alignment.cross_task import fit_material_entries, fit_material_root
from probe_transfer.alignment.materials import paired_split
from probe_transfer.alignment.methods import AlignmentMap, fit_ambient_alignments
from probe_transfer.alignment.ridge import RidgeSystem


def validate_alignment_selection(config: dict[str, Any]) -> None:
    settings = config["alignment"].get("fitting")
    if settings is None:
        return
    if not isinstance(settings, dict) or set(settings) != {"weighting", "relative_alphas"}:
        raise ConfigError("Alignment fitting requires weighting and relative_alphas only.")
    if settings["weighting"] not in {"uniform", "source_variance"}:
        raise ConfigError("Alignment weighting must be uniform or source_variance.")
    alphas = settings["relative_alphas"]
    if (
        not isinstance(alphas, list)
        or not alphas
        or any(
            type(value) not in {int, float} or not isfinite(value) or value <= 0 for value in alphas
        )
        or alphas != sorted(set(alphas))
    ):
        raise ConfigError("Relative alpha candidates must be finite, positive, unique, and sorted.")
    if config["alignment"]["methods"] != ["affine_ridge"] or not config.get("fit_materials"):
        raise ConfigError("Grouped selection requires affine ridge and explicit fit materials.")
    entries = fit_material_entries(config)
    if len(entries) < 2 or any(
        type(entry.get("expected_validation_rows")) is not int
        or entry["expected_validation_rows"] < 2
        for entry in entries
    ):
        raise ConfigError("Grouped selection requires at least two validation task contracts.")


def fit_configured_alignments(
    source_values: np.ndarray,
    target_values: np.ndarray,
    config: dict[str, Any],
    fit_root: Path,
    *,
    source: str,
    target: str,
    data_seed: int,
    layer: str,
    shuffle_seed: int,
    device: str,
) -> tuple[dict[str, AlignmentMap], list[dict[str, Any]]]:
    alignment = config["alignment"]
    settings = alignment.get("fitting")
    if settings is None:
        methods = [
            name
            for name in [*alignment["methods"], alignment["negative_control"]]
            if name != "quotient_ridge"
        ]
        return fit_ambient_alignments(
            source_values,
            target_values,
            relative_alpha=alignment["ridge_relative_alpha"],
            shuffle_seed=shuffle_seed,
            device=device,
            methods=methods,
        ), []
    validate_alignment_selection(config)
    validation, sizes = {}, {}
    for entry in fit_material_entries(config):
        root = fit_material_root(config, fit_root, entry)
        split = alignment.get("diagnostic_split", "validation")
        values = paired_split(root, source, target, f"seed_{data_seed}_{split}", layer)
        if any(len(value) != entry["expected_validation_rows"] for value in values):
            raise ValueError("Fitting-task validation rows do not match the contract.")
        name = entry["dataset_key"]
        validation[name] = (values[0], values[1])
        sizes[name] = entry.get("fit_rows", entry["expected_train_rows"])
    return fit_grouped_ridge(
        source_values,
        target_values,
        sizes,
        validation,
        weighting=settings["weighting"],
        relative_alphas=settings["relative_alphas"],
        shuffle_seed=alignment["shuffled_pairing_seed"]
        + 1000 * data_seed
        + sorted(config["models"]).index(source),
        device=device,
    )


def fit_grouped_ridge(
    source: np.ndarray,
    target: np.ndarray,
    group_sizes: dict[str, int],
    validation: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    weighting: str,
    relative_alphas: list[float],
    shuffle_seed: int,
    device: str,
) -> tuple[dict[str, AlignmentMap], list[dict[str, Any]]]:
    if source.ndim != 2 or source.shape != target.shape or sum(group_sizes.values()) != len(source):
        raise ValueError("Grouped ridge requires paired arrays matching the declared task sizes.")
    if set(validation) != set(group_sizes) or len(group_sizes) < 2 or min(group_sizes.values()) < 2:
        raise ValueError("Grouped ridge requires matching train and validation tasks.")
    source_tensor = torch.as_tensor(source, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    groups, stats, weights = [], [], []
    start = 0
    for name, size in group_sizes.items():
        stop = start + size
        train_source, train_target = source_tensor[start:stop], target_tensor[start:stop]
        val_source, val_target = [
            torch.as_tensor(array, dtype=torch.float32, device=device) for array in validation[name]
        ]
        if (
            val_source.ndim != 2
            or val_source.shape != val_target.shape
            or val_source.shape[1] != source.shape[1]
            or len(val_source) < 2
        ):
            raise ValueError("Validation activations must be paired with the training width.")
        arrays = (train_source, train_target, val_source, val_target)
        if any(not torch.isfinite(array).all() for array in arrays):
            raise ValueError("Alignment activations must be finite.")
        variance = float((train_source - train_source.mean(0)).square().mean().item())
        if not isfinite(variance) or variance <= torch.finfo(torch.float32).eps:
            raise ValueError("Task variance is too small for normalized selection.")
        sample_weight = 1.0 if weighting == "uniform" else 1.0 / variance
        if weighting not in {"uniform", "source_variance"}:
            raise ValueError("Unsupported task weighting.")
        weights.append(torch.full((size,), sample_weight, device=device))
        groups.append((train_source, train_target, val_source, val_target))
        stats.append(
            {
                "fit_task": name,
                "fit_rows": size,
                "validation_rows": len(val_source),
                "source_variance": variance,
                "source_rms": float(train_source.square().mean().sqrt().item()),
                "target_rms": float(train_target.square().mean().sqrt().item()),
                "target_variance": float(
                    (train_target - train_target.mean(0)).square().mean().item()
                ),
            }
        )
        start = stop
    row_weights = torch.cat(weights)
    row_weights /= row_weights.mean()
    start = 0
    for stat in stats:
        stat["sample_weight"] = float(row_weights[start].item())
        start += int(stat["fit_rows"])
    generator = torch.Generator().manual_seed(shuffle_seed)
    shuffled, start = [], 0
    for size in group_sizes.values():
        shuffled.append(torch.randperm(size, generator=generator) + start)
        start += size
    order = torch.cat(shuffled).to(device)
    fitted, records = {}, []
    for method, outputs in (
        ("affine_ridge", source_tensor),
        ("shuffled_affine_ridge", source_tensor.index_select(0, order)),
    ):
        system = RidgeSystem.prepare(
            outputs, target_tensor, None if weighting == "uniform" else row_weights
        )
        fitting_groups = [
            (part, *group[1:])
            for part, group in zip(outputs.split(list(group_sizes.values())), groups, strict=True)
        ]
        candidates = []
        for alpha in relative_alphas:
            weight, bias, penalty = system.solve(alpha)
            errors = [
                _errors(weight, bias, group, stat)
                for group, stat in zip(fitting_groups, stats, strict=True)
            ]
            objective = max(row["validation_relative_mse"] for row in errors)
            total_loss = sum(row["weighted_train_loss"] for row in errors)
            candidate = AlignmentMap(
                method,
                weight=weight,
                bias=bias,
                metadata={
                    "ridge_penalty": penalty,
                    "ridge_relative_alpha": alpha,
                    "selection_max_relative_mse": objective,
                },
            )
            candidates.append((objective, -alpha, candidate))
            for stat, error in zip(stats, errors, strict=True):
                records.append(
                    {
                        **stat,
                        **error,
                        "method": method,
                        "weighting": weighting,
                        "relative_alpha": alpha,
                        "selection_max_relative_mse": objective,
                        "weighted_train_loss_fraction": error["weighted_train_loss"]
                        / max(total_loss, 1e-30),
                    }
                )
        if not candidates:
            raise ValueError("At least one ridge candidate is required.")
        selected = min(candidates, key=lambda item: (item[0], item[1]))[2]
        fitted[method] = selected
        signature = _fingerprint(selected)
        for row in records:
            if row["method"] == method:
                row["selected"] = row["relative_alpha"] == selected.metadata["ridge_relative_alpha"]
                if row["selected"]:
                    row["map_fingerprint"] = signature
    return fitted, records


def _errors(weight, bias, arrays, stat) -> dict[str, float]:
    train_source, train_target, val_source, val_target = arrays
    train_mse = float((train_target @ weight + bias - train_source).square().mean().item())
    validation_mse = float((val_target @ weight + bias - val_source).square().mean().item())
    if not isfinite(train_mse) or not isfinite(validation_mse):
        raise ValueError("Non-finite ridge reconstruction error.")
    return {
        "train_mse": train_mse,
        "validation_mse": validation_mse,
        "train_relative_mse": train_mse / stat["source_variance"],
        "validation_relative_mse": validation_mse / stat["source_variance"],
        "weighted_train_loss": train_mse * stat["fit_rows"] * stat["sample_weight"],
    }


def _fingerprint(alignment: AlignmentMap) -> str:
    digest = sha256()
    for tensor in (alignment.weight, alignment.bias):
        if tensor is None:
            raise ValueError("A selected affine map must contain weights and bias.")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
