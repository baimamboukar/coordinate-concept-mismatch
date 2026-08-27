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
    fit_positive_diagonal_alignment,
)
from probe_transfer.extraction.activations import load_activation_split
from probe_transfer.symmetry.coordinates import CoordinateTransform
from probe_transfer.symmetry.protocol import estimated_alignment_enabled, selected_models

MapKey = tuple[int, str, float, int]


def estimate_transformation_maps(
    baseline_dir: Path,
    config: dict[str, Any],
    transformations: dict[int, CoordinateTransform],
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
                for transformation_seed, transformation in transformations.items():
                    target_fit = transformation.apply_array(source_fit)
                    if settings["method"] == "exact_permutation":
                        fitted = fit_exact_permutation_alignment(source_fit, target_fit)
                    elif settings["method"] == "permutation":
                        fitted = fit_permutation_alignment(
                            source_fit,
                            target_fit,
                            device=device,
                        )
                    else:
                        fitted = fit_positive_diagonal_alignment(
                            source_fit,
                            target_fit,
                            relative_tolerance=settings["fit_relative_tolerance"],
                        )
                    key = (data_seed, model, depth, transformation_seed)
                    maps[key] = fitted
                    diagnostics.append(
                        _diagnostic(
                            fitted,
                            validation,
                            transformation,
                            data_seed,
                            model,
                            depth,
                            transformation_seed,
                            fit_rows,
                            settings.get("scale_match_rtol", 0.0),
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
    transformation: CoordinateTransform,
    data_seed: int,
    model: str,
    depth: float,
    transformation_seed: int,
    fit_rows: int,
    scale_match_rtol: float,
) -> dict[str, Any]:
    target = transformation.apply_array(source)
    record = {
        "data_seed": data_seed,
        "model": model,
        "depth": depth,
        "transformation_seed": transformation_seed,
        "method": fitted.method,
        "fit_rows": fit_rows,
        **fitted.metadata,
        **alignment_diagnostic(fitted, source, target),
    }
    if transformation.kind == "permutation":
        if fitted.indices is None:
            raise RuntimeError("Estimated permutation is missing feature indices.")
        expected = transformation.inverse().values.cpu()
        actual = fitted.indices.detach().cpu()
        record["permutation_match_fraction"] = float((actual == expected).float().mean().item())
    else:
        if fitted.scale is None or scale_match_rtol <= 0:
            raise RuntimeError("Estimated positive diagonal is missing scales or tolerance.")
        expected = transformation.inverse().values.float().cpu()
        actual = fitted.scale.detach().float().cpu()
        relative = torch.abs(actual - expected) / expected
        record["scale_match_fraction"] = float(
            torch.isclose(actual, expected, rtol=scale_match_rtol, atol=0).float().mean().item()
        )
        record["maximum_scale_relative_error"] = float(relative.max().item())
    return record
