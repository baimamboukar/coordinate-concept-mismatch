from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

from probe_transfer.alignment.cross_task import fit_material_entries, fit_material_root
from probe_transfer.alignment.methods import AlignmentMap
from probe_transfer.alignment.quotient import effective_linear_parameters
from probe_transfer.alignment.ridge import RidgeSystem
from probe_transfer.probes.transport import load_probe_bundle

Group = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
ProbeParameters = tuple[np.ndarray, float]


def load_fit_probe_bank(
    config: dict[str, Any], root: Path, source: str, data_seed: int, layer: str
) -> dict[str, ProbeParameters]:
    probes = {}
    for entry in fit_material_entries(config):
        bundle = (
            fit_material_root(config, root, entry)
            / "probes"
            / f"seed_{data_seed}"
            / f"{source}.safetensors"
        )
        key = f"{layer}.linear"
        loaded = load_probe_bundle(bundle)
        if key not in loaded:
            raise ValueError(f"Fit probe bundle is missing {key}: {bundle}")
        probes[entry["dataset_key"]] = effective_linear_parameters(loaded[key])
    return probes


def fit_probe_bank_alignment(
    base: AlignmentMap,
    groups: dict[str, Group],
    probes: dict[str, ProbeParameters],
    stats: dict[str, dict[str, float | int | str]],
    relative_alphas: list[float],
    variance_power: float,
    weighting: str,
) -> tuple[AlignmentMap, list[dict[str, Any]]]:
    if (
        base.weight is None
        or base.bias is None
        or set(groups) != set(probes)
        or set(groups) != set(stats)
    ):
        raise ValueError("Probe-bank alignment requires complete task-aligned materials.")

    selected, candidates = {}, {}
    for name, group in groups.items():
        train_source, train_target, validation_source, validation_target = group
        weight, intercept = probes[name]
        probe_weight = torch.as_tensor(weight, dtype=train_source.dtype, device=train_source.device)
        expected_train = train_source @ probe_weight + intercept
        expected_validation = validation_source @ probe_weight + intercept
        denominator = expected_train.var(unbiased=False).clamp_min(
            torch.finfo(train_source.dtype).eps
        )
        system = RidgeSystem.prepare(expected_train[:, None], train_target)
        task_candidates = []
        for alpha in relative_alphas:
            fitted_weight, fitted_bias, penalty = system.solve(alpha)
            predicted = (validation_target @ fitted_weight + fitted_bias).squeeze(1)
            error = float(((predicted - expected_validation).square().mean() / denominator).item())
            task_candidates.append((error, -alpha, fitted_weight[:, 0], fitted_bias[0], penalty))
        chosen = min(task_candidates, key=lambda item: (item[0], item[1]))
        selected[name] = chosen
        candidates[name] = task_candidates

    source_weights = torch.stack(
        [
            torch.as_tensor(probes[name][0], dtype=base.weight.dtype, device=base.weight.device)
            for name in groups
        ]
    )
    if int(torch.linalg.matrix_rank(source_weights).item()) != len(source_weights):
        raise ValueError("Probe-bank directions must be linearly independent.")
    target_weights = torch.stack([selected[name][2] for name in groups])
    target_biases = torch.stack([selected[name][3] for name in groups])
    source_biases = torch.tensor(
        [probes[name][1] for name in groups], dtype=base.bias.dtype, device=base.bias.device
    )
    inverse = torch.linalg.pinv(source_weights)
    lifted = base.weight.T + inverse @ (target_weights - source_weights @ base.weight.T)
    lifted_bias = base.bias + inverse @ (target_biases - source_biases - source_weights @ base.bias)
    maximum_error = max(selected[name][0] for name in groups)
    fitted = AlignmentMap(
        "probe_bank_affine",
        weight=lifted.T,
        bias=lifted_bias,
        metadata={
            "probe_bank_size": len(source_weights),
            "source_variance_power": variance_power,
            "selection_max_probe_score_relative_mse": maximum_error,
        },
    )

    signature = _fingerprint(fitted)
    records = []
    feature_errors = {
        name: _feature_errors(fitted, group, stats[name]) for name, group in groups.items()
    }
    total = sum(values["weighted_train_loss"] for values in feature_errors.values())
    for name, task_candidates in candidates.items():
        chosen_alpha = -selected[name][1]
        for error, negative_alpha, _, _, penalty in task_candidates:
            alpha = -negative_alpha
            row = {
                **stats[name],
                **feature_errors[name],
                "method": "probe_bank_affine",
                "weighting": weighting,
                "source_variance_power": variance_power,
                "selection_metric": "worst_probe_score_mse",
                "relative_alpha": alpha,
                "ridge_penalty": penalty,
                "candidate_probe_score_relative_mse": error,
                "selection_max_relative_mse": maximum_error,
                "weighted_train_loss_fraction": feature_errors[name]["weighted_train_loss"]
                / max(total, 1e-30),
                "selected": alpha == chosen_alpha,
            }
            if row["selected"]:
                row["map_fingerprint"] = signature
            records.append(row)
    return fitted, records


def _feature_errors(
    alignment: AlignmentMap, group: Group, stat: dict[str, float | int | str]
) -> dict[str, float]:
    if alignment.weight is None or alignment.bias is None:
        raise ValueError("Probe-bank alignment is incomplete.")
    train_source, train_target, validation_source, validation_target = group
    train_mse = float(
        (train_target @ alignment.weight + alignment.bias - train_source).square().mean().item()
    )
    validation_mse = float(
        (validation_target @ alignment.weight + alignment.bias - validation_source)
        .square()
        .mean()
        .item()
    )
    source_variance = float(stat["source_variance"])
    return {
        "train_mse": train_mse,
        "validation_mse": validation_mse,
        "train_relative_mse": train_mse / source_variance,
        "validation_relative_mse": validation_mse / source_variance,
        "weighted_train_loss": train_mse * int(stat["fit_rows"]) * float(stat["sample_weight"]),
    }


def _fingerprint(alignment: AlignmentMap) -> str:
    digest = sha256()
    for tensor in (alignment.weight, alignment.bias):
        if tensor is None:
            raise ValueError("A probe-bank map must contain weights and bias.")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
