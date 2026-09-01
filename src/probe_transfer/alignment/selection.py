from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from core.config import ConfigError
from probe_transfer.alignment.cross_task import fit_material_entries, fit_material_root
from probe_transfer.alignment.grouped_ridge import fit_grouped_ridge
from probe_transfer.alignment.materials import paired_split
from probe_transfer.alignment.methods import AlignmentMap, fit_ambient_alignments
from probe_transfer.alignment.probe_bank import load_fit_probe_bank


def validate_alignment_selection(config: dict[str, Any]) -> None:
    settings = config["alignment"].get("fitting")
    if settings is None:
        return
    required = {"weighting", "relative_alphas"}
    allowed = required | {"source_variance_powers", "selection_metric"}
    if not isinstance(settings, dict) or not required <= set(settings) or set(settings) - allowed:
        raise ConfigError(
            "Alignment fitting requires a supported grouped-ridge selection contract."
        )
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
    _variance_powers(settings)
    metric = settings.get("selection_metric", "worst_feature_mse")
    if metric not in {"worst_feature_mse", "worst_probe_score_mse"}:
        raise ConfigError("Unsupported grouped-ridge selection metric.")
    methods = config["alignment"]["methods"]
    if (
        "affine_ridge" not in methods
        or set(methods) - {"affine_ridge", "probe_bank_affine"}
        or not config.get("fit_materials")
    ):
        raise ConfigError("Grouped selection requires affine ridge and explicit fit materials.")
    entries = fit_material_entries(config)
    if len(entries) < 2 or any(
        type(entry.get("expected_validation_rows")) is not int
        or entry["expected_validation_rows"] < 2
        for entry in entries
    ):
        raise ConfigError("Grouped selection requires at least two validation task contracts.")
    if "probe_bank_affine" in methods or metric == "worst_probe_score_mse":
        for entry in entries:
            for key in ("probe_source_name", "probe_source_variant"):
                if not isinstance(entry.get(key), str) or not entry[key]:
                    raise ConfigError(
                        "Probe-aware fitting requires pinned fit-task probe materials."
                    )


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
    requires_probes = (
        "probe_bank_affine" in alignment["methods"]
        or settings.get("selection_metric") == "worst_probe_score_mse"
    )
    probes = (
        load_fit_probe_bank(config, fit_root, source, data_seed, layer) if requires_probes else None
    )
    return fit_grouped_ridge(
        source_values,
        target_values,
        sizes,
        validation,
        weighting=settings["weighting"],
        relative_alphas=settings["relative_alphas"],
        source_variance_powers=_variance_powers(settings),
        selection_metric=settings.get("selection_metric", "worst_feature_mse"),
        probe_parameters=probes,
        include_probe_bank="probe_bank_affine" in alignment["methods"],
        shuffle_seed=alignment["shuffled_pairing_seed"]
        + 1000 * data_seed
        + sorted(config["models"]).index(source),
        device=device,
    )


def _variance_powers(settings: dict[str, Any]) -> list[float]:
    configured = settings.get("source_variance_powers")
    if configured is None:
        return [0.0] if settings["weighting"] == "uniform" else [1.0]
    if settings["weighting"] != "source_variance":
        raise ConfigError("Variance-power candidates require source_variance weighting.")
    if (
        not isinstance(configured, list)
        or not configured
        or any(
            type(value) not in {int, float} or not isfinite(value) or not 0 <= value <= 1
            for value in configured
        )
        or configured != sorted(set(configured))
    ):
        raise ConfigError("Variance powers must be unique sorted values between zero and one.")
    return [float(value) for value in configured]
