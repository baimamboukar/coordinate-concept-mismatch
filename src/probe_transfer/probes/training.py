from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from probe_transfer.probes.models import (
    CPDegree2,
    LinearProbe,
    NeuralProbe,
    OneHiddenLayerMLP,
    train_linear_probe,
    train_neural_probe,
)

Probe = LinearProbe | NeuralProbe


@dataclass(frozen=True)
class ProbeSelection:
    probe: Probe
    parameters: dict[str, Any]


def families_for_depth(probes: dict[str, Any], depth: float, primary_depth: float) -> list[str]:
    key = "primary_families" if depth == primary_depth else "secondary_families"
    families = probes[key]
    if not families or len(families) != len(set(families)):
        raise ValueError(f"{key} must contain unique probe families.")
    return list(families)


def fit_probe_family(
    family: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config: dict[str, Any],
    *,
    device: str,
) -> ProbeSelection:
    if family == "linear":
        spec = config[family]
        probe = train_linear_probe(
            train_x,
            train_y,
            validation_x,
            validation_y,
            c_values=spec["c_values"],
            max_iter=spec["max_iter"],
        )
        return ProbeSelection(probe, {"c": probe.c})
    if family not in {"cp_degree_2", "mlp"}:
        raise ValueError(f"Unsupported probe family: {family}")
    return _fit_neural_family(
        family,
        train_x,
        train_y,
        validation_x,
        validation_y,
        config[family],
        device=device,
    )


def _fit_neural_family(
    family: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    spec: dict[str, Any],
    *,
    device: str,
) -> ProbeSelection:
    candidates: list[tuple[tuple[float, float, float, float], ProbeSelection]] = []
    capacities = spec["ranks"] if family == "cp_degree_2" else [spec["hidden_size"]]
    restart_seeds = spec["restart_seeds"]
    if not capacities or not spec["weight_decays"] or not restart_seeds:
        raise ValueError(f"{family} requires capacities, weight decays, and restart seeds.")

    for capacity in capacities:
        for weight_decay in spec["weight_decays"]:
            for restart_seed in restart_seeds:
                factory = _model_factory(family, int(capacity))
                probe = train_neural_probe(
                    factory,
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                    learning_rate=spec["learning_rate"],
                    weight_decay=float(weight_decay),
                    batch_size=spec["batch_size"],
                    max_epochs=spec["max_epochs"],
                    patience=spec["patience"],
                    seed=int(restart_seed),
                    device=device,
                )
                parameters = {
                    "capacity": int(capacity),
                    "weight_decay": float(weight_decay),
                    "restart_seed": int(restart_seed),
                    "epochs": probe.epochs,
                }
                tie_break = (
                    probe.validation_auroc,
                    -float(capacity),
                    -float(weight_decay),
                    -float(restart_seed),
                )
                candidates.append((tie_break, ProbeSelection(probe, parameters)))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return max(candidates, key=lambda candidate: candidate[0])[1]


def _model_factory(family: str, capacity: int):
    if family == "cp_degree_2":
        return lambda input_size: CPDegree2(input_size, rank=capacity)
    return lambda input_size: OneHiddenLayerMLP(input_size, hidden_size=capacity)
