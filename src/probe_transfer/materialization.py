import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from core.constants import HF_TOKEN_ENVIRONMENTS
from probe_transfer.alignment.cross_task import fit_material_entries, fit_material_root
from probe_transfer.data import load_prepared_rows
from probe_transfer.layout import (
    activation_prefix,
    artifact_uri,
    study_prefix,
)
from probe_transfer.transfer.runner import _validate_metadata


def verify_prior_probe_splits(config: dict[str, Any], rows: Path, destination: Path) -> int:
    reference = config["artifacts"].get("split_reference")
    if reference is None:
        return 0
    model = reference["model"]
    names = [
        f"seed_{seed}_{split}" for seed in config["data_seeds"] for split in ("train", "validation")
    ]
    _sync_public(
        config,
        activation_prefix(config, model, dataset_key=reference["dataset_key"]),
        destination,
        include=[f"{name}.safetensors" for name in names],
    )
    for name in names:
        count = config["sampling"][f"{name.rsplit('_', 1)[1]}_size"]
        prepared = load_prepared_rows(rows / f"{name}.jsonl", count)
        with safe_open(destination / f"{name}.safetensors", framework="pt", device="cpu") as saved:
            _validate_metadata(saved.metadata(), config["models"][model], name, count, config)
            for field, column in (("row_ids", "row_id"), ("labels", "label")):
                if not torch.equal(
                    saved.get_tensor(field), torch.tensor([row[column] for row in prepared])
                ):
                    raise ValueError(
                        f"Prior probe {column} order differs from the pinned split: {name}"
                    )
    return len(names)


def materialize_activations(
    config: dict[str, Any], destination: Path, models: list[str] | None = None
) -> None:
    for model in models or list(config["models"]):
        _sync_public(
            config,
            activation_prefix(config, model),
            destination / "activations" / model,
        )


def materialize_fit_activations(config: dict[str, Any], destination: Path) -> None:
    for fit in fit_material_entries(config):
        root = fit_material_root(config, destination, fit)
        for model in config["models"]:
            _sync_public(
                config,
                activation_prefix(config, model, dataset_key=fit["dataset_key"]),
                root / "activations" / model,
            )


def materialize_recovery_reference(config: dict[str, Any], destination: Path) -> Path:
    reference = config["reference_materials"]
    source = study_prefix(
        reference["source_name"],
        reference["source_study"],
        reference["source_variant"],
    )
    _sync_public(
        config,
        f"{source}/results",
        destination / "results",
        include="recovery.jsonl",
    )
    return destination / "results" / "recovery.jsonl"


def materialize_baseline(
    config: dict[str, Any], destination: Path, models: list[str] | None = None
) -> None:
    materials = config["materials"]
    source = study_prefix(
        materials["source_name"],
        materials["source_study"],
        materials.get("source_variant"),
    )
    includes = None
    if models is not None:
        includes = [
            f"seed_{seed}/{model}.safetensors" for seed in config["data_seeds"] for model in models
        ]
    _sync_public(config, f"{source}/probes", destination / "probes", include=includes)
    _sync_public(
        config,
        f"{source}/results",
        destination / "results",
        include="metrics.jsonl",
    )
    materialize_activations(config, destination, models=models)


def _sync_public(
    config: dict[str, Any],
    remote_prefix: str,
    destination: Path,
    include: str | list[str] | None = None,
) -> None:
    hf = shutil.which("hf")
    if hf is None:
        raise RuntimeError("The Hugging Face CLI is required to materialize worker inputs.")
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        hf,
        "buckets",
        "sync",
        artifact_uri(config, remote_prefix),
        str(destination),
        "--no-delete",
        "--quiet",
    ]
    for pattern in [include] if isinstance(include, str) else include or []:
        command.extend(["--include", pattern])
    environment = os.environ.copy()
    for name in HF_TOKEN_ENVIRONMENTS:
        environment.pop(name, None)
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    subprocess.run(command, check=True, env=environment)
