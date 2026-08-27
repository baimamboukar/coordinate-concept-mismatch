from pathlib import Path
from typing import Any

import numpy as np
import torch

from probe_transfer.alignment.materials import layer_key, resolve_device
from probe_transfer.alignment.methods import (
    AlignmentMap,
    alignment_diagnostic,
    fit_exact_permutation_alignment,
    fit_permutation_alignment,
)
from probe_transfer.extraction.activations import load_activation_split
from probe_transfer.symmetry.protocol import estimated_alignment_enabled, selected_models
from probe_transfer.symmetry.transforms import inverse_permutation

MapKey = tuple[int, str, float, int]


def estimate_permutation_maps(
    baseline_dir: Path,
    config: dict[str, Any],
    permutations: dict[int, torch.Tensor],
) -> tuple[dict[MapKey, AlignmentMap], list[dict[str, Any]]]:
    if not estimated_alignment_enabled(config):
        return {}, []

    settings = config["symmetry"]["estimated_alignment"]
    device = resolve_device(settings["device"])
    fit_rows = settings["fit_rows"]
    maps: dict[MapKey, AlignmentMap] = {}
    diagnostics: list[dict[str, Any]] = []

    for data_seed in config["data_seeds"]:
        for model in selected_models(config):
            model_dir = baseline_dir / "activations" / model
            for depth in config["symmetry"]["probed_depths"]:
                layer = layer_key(depth)
                train = _activations(model_dir / f"seed_{data_seed}_train.safetensors", layer)
                validation = _activations(
                    model_dir / f"seed_{data_seed}_validation.safetensors", layer
                )
                _require_rows(train, config["materials"]["expected_train_rows"], "training")
                _require_rows(
                    validation,
                    config["materials"]["expected_validation_rows"],
                    "validation",
                )
                source_fit = train[:fit_rows]
                for permutation_seed, permutation in permutations.items():
                    index = permutation.numpy()
                    target_fit = source_fit[:, index]
                    if settings["method"] == "exact_permutation":
                        fitted = fit_exact_permutation_alignment(source_fit, target_fit)
                    else:
                        fitted = fit_permutation_alignment(
                            source_fit,
                            target_fit,
                            device=device,
                        )
                    key = (data_seed, model, depth, permutation_seed)
                    maps[key] = fitted
                    diagnostics.append(
                        _diagnostic(
                            fitted,
                            validation,
                            permutation,
                            data_seed,
                            model,
                            depth,
                            permutation_seed,
                            fit_rows,
                        )
                    )
    return maps, diagnostics


def _activations(path: Path, layer: str) -> np.ndarray:
    values, _, _ = load_activation_split(path, layer)
    return values.numpy()


def _require_rows(values: np.ndarray, expected: int, split: str) -> None:
    if len(values) != expected:
        raise ValueError(f"Expected {expected} {split} activations, found {len(values)}.")


def _diagnostic(
    fitted: AlignmentMap,
    source: np.ndarray,
    permutation: torch.Tensor,
    data_seed: int,
    model: str,
    depth: float,
    permutation_seed: int,
    fit_rows: int,
) -> dict[str, Any]:
    if fitted.indices is None:
        raise RuntimeError("Estimated permutation is missing feature indices.")
    expected = inverse_permutation(permutation).cpu()
    actual = fitted.indices.detach().cpu()
    return {
        "data_seed": data_seed,
        "model": model,
        "depth": depth,
        "permutation_seed": permutation_seed,
        "method": fitted.method,
        "fit_rows": fit_rows,
        "permutation_match_fraction": float((actual == expected).float().mean().item()),
        **fitted.metadata,
        **alignment_diagnostic(fitted, source, source[:, permutation.numpy()]),
    }
