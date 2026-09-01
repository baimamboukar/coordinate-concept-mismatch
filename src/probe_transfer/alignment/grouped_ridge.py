from hashlib import sha256
from math import isfinite
from typing import Any

import numpy as np
import torch

from probe_transfer.alignment.methods import AlignmentMap
from probe_transfer.alignment.probe_bank import (
    ProbeParameters,
    fit_probe_bank_alignment,
)
from probe_transfer.alignment.ridge import RidgeSystem

Group = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def fit_grouped_ridge(
    source: np.ndarray,
    target: np.ndarray,
    group_sizes: dict[str, int],
    validation: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    weighting: str,
    relative_alphas: list[float],
    source_variance_powers: list[float] | None = None,
    selection_metric: str = "worst_feature_mse",
    probe_parameters: dict[str, ProbeParameters] | None = None,
    include_probe_bank: bool = False,
    shuffle_seed: int,
    device: str,
) -> tuple[dict[str, AlignmentMap], list[dict[str, Any]]]:
    _validate_inputs(source, target, group_sizes, validation, weighting)
    source_values = torch.as_tensor(source, dtype=torch.float32, device=device)
    target_values = torch.as_tensor(target, dtype=torch.float32, device=device)
    groups, stats = _groups(source_values, target_values, group_sizes, validation, device)
    if selection_metric == "worst_probe_score_mse" and set(probe_parameters or {}) != set(groups):
        raise ValueError("Probe-score selection requires one source probe per fitting task.")
    powers = source_variance_powers or ([0.0] if weighting == "uniform" else [1.0])
    order = _shuffled_order(group_sizes, shuffle_seed, device)
    fitted, records = {}, []
    for method, outputs in (
        ("affine_ridge", source_values),
        ("shuffled_affine_ridge", source_values.index_select(0, order)),
    ):
        selected, method_records = _select_map(
            method,
            outputs,
            target_values,
            groups,
            stats,
            weighting,
            powers,
            relative_alphas,
            selection_metric,
            probe_parameters,
        )
        fitted[method] = selected
        records.extend(method_records)
    if include_probe_bank:
        if set(probe_parameters or {}) != set(groups):
            raise ValueError("Probe-bank alignment requires one source probe per fitting task.")
        power = float(fitted["affine_ridge"].metadata["source_variance_power"])
        _, selected_stats = _weighted_stats(stats, power, device)
        bank, bank_records = fit_probe_bank_alignment(
            fitted["affine_ridge"],
            groups,
            probe_parameters or {},
            selected_stats,
            relative_alphas,
            power,
            weighting,
        )
        fitted[bank.method] = bank
        records.extend(bank_records)
    return fitted, records


def _select_map(method, outputs, target, groups, stats, weighting, powers, alphas, metric, probes):
    records, candidates = [], []
    sizes = [len(group[0]) for group in groups.values()]
    fitting = {
        name: (part, *groups[name][1:])
        for name, part in zip(groups, outputs.split(sizes), strict=True)
    }
    for power in powers:
        weights, weighted_stats = _weighted_stats(stats, power, target.device)
        system = RidgeSystem.prepare(outputs, target, None if power == 0 else weights)
        for alpha in alphas:
            weight, bias, penalty = system.solve(alpha)
            errors = {
                name: _errors(
                    weight,
                    bias,
                    fitting[name],
                    weighted_stats[name],
                    None if probes is None else probes[name],
                )
                for name in groups
            }
            field = (
                "validation_probe_score_relative_mse"
                if metric == "worst_probe_score_mse"
                else "validation_relative_mse"
            )
            objective = max(float(errors[name][field]) for name in groups)
            total = sum(float(row["weighted_train_loss"]) for row in errors.values())
            candidate = AlignmentMap(
                method,
                weight=weight,
                bias=bias,
                metadata={
                    "ridge_penalty": penalty,
                    "ridge_relative_alpha": alpha,
                    "source_variance_power": power,
                    "selection_max_relative_mse": objective,
                },
            )
            candidates.append((objective, -alpha, power, candidate))
            for name in groups:
                records.append(
                    {
                        **weighted_stats[name],
                        **errors[name],
                        "method": method,
                        "weighting": weighting,
                        "source_variance_power": power,
                        "selection_metric": metric,
                        "relative_alpha": alpha,
                        "selection_max_relative_mse": objective,
                        "weighted_train_loss_fraction": float(errors[name]["weighted_train_loss"])
                        / max(total, 1e-30),
                    }
                )
    selected = min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]
    signature = _fingerprint(selected)
    for row in records:
        row["selected"] = (
            row["relative_alpha"] == selected.metadata["ridge_relative_alpha"]
            and row["source_variance_power"] == selected.metadata["source_variance_power"]
        )
        if row["selected"]:
            row["map_fingerprint"] = signature
    return selected, records


def _groups(source, target, sizes, validation, device):
    groups, stats, start = {}, {}, 0
    for name, size in sizes.items():
        stop = start + size
        val_source, val_target = [
            torch.as_tensor(array, dtype=torch.float32, device=device) for array in validation[name]
        ]
        group = (source[start:stop], target[start:stop], val_source, val_target)
        if val_source.shape != val_target.shape or val_source.shape[1] != source.shape[1]:
            raise ValueError("Validation activations must be paired and width-matched.")
        if any(not torch.isfinite(array).all() for array in group):
            raise ValueError("Alignment activations must be finite.")
        variance = float((group[0] - group[0].mean(0)).square().mean().item())
        if not isfinite(variance) or variance <= torch.finfo(torch.float32).eps:
            raise ValueError("Task variance is too small for normalized selection.")
        groups[name] = group
        stats[name] = {
            "fit_task": name,
            "fit_rows": size,
            "validation_rows": len(val_source),
            "source_variance": variance,
            "source_rms": float(group[0].square().mean().sqrt().item()),
            "target_rms": float(group[1].square().mean().sqrt().item()),
            "target_variance": float((group[1] - group[1].mean(0)).square().mean().item()),
        }
        start = stop
    return groups, stats


def _errors(weight, bias, group: Group, stat, probe):
    train_source, train_target, val_source, val_target = group
    train_mse = float((train_target @ weight + bias - train_source).square().mean().item())
    validation_mse = float((val_target @ weight + bias - val_source).square().mean().item())
    source_variance = float(stat["source_variance"])
    return {
        "train_mse": train_mse,
        "validation_mse": validation_mse,
        "train_relative_mse": train_mse / source_variance,
        "validation_relative_mse": validation_mse / source_variance,
        "validation_probe_score_relative_mse": (
            None if probe is None else _probe_score_error(weight, bias, group, probe)
        ),
        "weighted_train_loss": train_mse * int(stat["fit_rows"]) * float(stat["sample_weight"]),
    }


def _probe_score_error(weight, bias, group: Group, probe: ProbeParameters) -> float:
    train_source, _, val_source, val_target = group
    values, intercept = probe
    probe_weight = torch.as_tensor(values, dtype=train_source.dtype, device=train_source.device)
    train_scores = train_source @ probe_weight + intercept
    expected = val_source @ probe_weight + intercept
    predicted = (val_target @ weight + bias) @ probe_weight + intercept
    denominator = train_scores.var(unbiased=False).clamp_min(torch.finfo(train_scores.dtype).eps)
    return float(((predicted - expected).square().mean() / denominator).item())


def _weighted_stats(stats, power, device):
    weights = [
        torch.full(
            (int(stat["fit_rows"]),),
            float(stat["source_variance"]) ** -power,
            device=device,
        )
        for stat in stats.values()
    ]
    row_weights = torch.cat(weights)
    row_weights /= row_weights.mean()
    weighted, start = {}, 0
    for name, stat in stats.items():
        weighted[name] = {**stat, "sample_weight": float(row_weights[start].item())}
        start += int(stat["fit_rows"])
    return row_weights, weighted


def _validate_inputs(source, target, sizes, validation, weighting):
    if source.ndim != 2 or source.shape != target.shape or sum(sizes.values()) != len(source):
        raise ValueError("Grouped ridge requires paired arrays matching the declared task sizes.")
    if set(validation) != set(sizes) or len(sizes) < 2 or min(sizes.values()) < 2:
        raise ValueError("Grouped ridge requires matching train and validation tasks.")
    if weighting not in {"uniform", "source_variance"}:
        raise ValueError("Unsupported task weighting.")


def _shuffled_order(sizes, seed, device):
    generator = torch.Generator().manual_seed(seed)
    parts, start = [], 0
    for size in sizes.values():
        parts.append(torch.randperm(size, generator=generator) + start)
        start += size
    return torch.cat(parts).to(device)


def _fingerprint(alignment: AlignmentMap) -> str:
    digest = sha256()
    for tensor in (alignment.weight, alignment.bias):
        if tensor is None:
            raise ValueError("A selected affine map must contain weights and bias.")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
