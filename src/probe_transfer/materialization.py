import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.constants import HF_TOKEN_ENVIRONMENTS
from probe_transfer.layout import (
    activation_prefix,
    artifact_uri,
    study_prefix,
)


def materialize_activations(
    config: dict[str, Any], destination: Path, models: list[str] | None = None
) -> None:
    for model in models or list(config["models"]):
        _sync_public(
            config,
            activation_prefix(config, model),
            destination / "activations" / model,
        )


def materialize_baseline(config: dict[str, Any], destination: Path) -> None:
    materials = config["materials"]
    source = study_prefix(materials["source_name"], materials["source_study"])
    _sync_public(config, f"{source}/probes", destination / "probes")
    _sync_public(
        config,
        f"{source}/results",
        destination / "results",
        include="metrics.jsonl",
    )
    materialize_activations(config, destination)


def _sync_public(
    config: dict[str, Any], remote_prefix: str, destination: Path, include: str | None = None
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
    if include:
        command.extend(["--include", include])
    environment = os.environ.copy()
    for name in HF_TOKEN_ENVIRONMENTS:
        environment.pop(name, None)
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    subprocess.run(command, check=True, env=environment)
