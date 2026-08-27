import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional

from probe_transfer.artifacts import sha256_file
from probe_transfer.symmetry.coordinates import CoordinateTransform


@dataclass(frozen=True)
class StoredProbe:
    name: str
    kind: str
    tensors: dict[str, torch.Tensor]
    details: dict[str, Any]

    def scores(self, activations: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(activations, dtype=torch.float32)
        mean = self.tensors["preprocessor.mean"]
        scale = self.tensors["preprocessor.scale"]
        if scale.ndim != 0 or not torch.isfinite(scale) or scale <= 0:
            raise ValueError(f"Probe {self.name} has an invalid preprocessing scale.")
        inputs = (values - mean) / scale

        if self.kind == "linear":
            scores = functional.linear(
                inputs,
                self.tensors["coefficient"],
                self.tensors["intercept"],
            )
        elif self.kind == "CPDegree2":
            left = functional.linear(
                inputs,
                self.tensors["model.left.weight"],
                self.tensors["model.left.bias"],
            )
            right = functional.linear(
                inputs,
                self.tensors["model.right.weight"],
                self.tensors["model.right.bias"],
            )
            scores = (left * right * self.tensors["model.alpha"]).sum(dim=-1, keepdim=True)
            scores += functional.linear(
                inputs,
                self.tensors["model.linear.weight"],
                self.tensors["model.linear.bias"],
            )
        elif self.kind == "OneHiddenLayerMLP":
            hidden = functional.gelu(
                functional.linear(
                    inputs,
                    self.tensors["model.network.0.weight"],
                    self.tensors["model.network.0.bias"],
                )
            )
            scores = functional.linear(
                hidden,
                self.tensors["model.network.2.weight"],
                self.tensors["model.network.2.bias"],
            )
        else:
            raise ValueError(f"Unsupported stored probe kind: {self.kind}")
        return scores.squeeze(-1).numpy()

    def transport(self, coordinates: CoordinateTransform | torch.Tensor) -> "StoredProbe":
        if isinstance(coordinates, torch.Tensor):
            coordinates = CoordinateTransform("permutation", coordinates)
        if len(coordinates.values) != self.tensors["preprocessor.mean"].numel():
            raise ValueError(f"Probe {self.name} and coordinate widths do not match.")

        tensors = {name: value.clone() for name, value in self.tensors.items()}
        tensors["preprocessor.mean"] = coordinates.apply_tensor(tensors["preprocessor.mean"])
        for name in _input_weights(self.kind):
            if coordinates.kind == "permutation":
                tensors[name] = tensors[name].index_select(1, coordinates.values)
            else:
                scales = coordinates.values.to(tensors[name].dtype)
                tensors[name] = tensors[name] / scales[None, :]
        return StoredProbe(self.name, self.kind, tensors, dict(self.details))


def load_probe_bundle(path: str | Path) -> dict[str, StoredProbe]:
    with safe_open(path, framework="pt", device="cpu") as saved:
        metadata = saved.metadata() or {}
        raw_details = metadata.get("probes")
        if raw_details is None:
            raise ValueError("Probe bundle is missing its probes metadata.")
        details = json.loads(raw_details)
        tensors = {}
        for name in saved.keys():  # noqa: SIM118
            tensors[name] = saved.get_tensor(name).float()

    probes = {}
    for name, record in details.items():
        prefix = f"{name}."
        scoped = {
            key.removeprefix(prefix): value
            for key, value in tensors.items()
            if key.startswith(prefix)
        }
        if not scoped:
            raise ValueError(f"Probe bundle has no tensors for {name}.")
        probes[name] = StoredProbe(name, record["kind"], scoped, dict(record))
    return probes


def save_probe_bundle(
    path: str | Path,
    probes: dict[str, StoredProbe],
    *,
    metadata_updates: dict[str, Any] | None = None,
) -> str:
    tensors = {}
    metadata = {}
    for name, probe in sorted(probes.items()):
        tensors.update(
            {f"{name}.{key}": value.contiguous() for key, value in probe.tensors.items()}
        )
        metadata[name] = {**probe.details, **(metadata_updates or {})}

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output, metadata={"probes": json.dumps(metadata, sort_keys=True)})
    return sha256_file(output)


def _input_weights(kind: str) -> tuple[str, ...]:
    if kind == "linear":
        return ("coefficient",)
    if kind == "CPDegree2":
        return ("model.left.weight", "model.right.weight", "model.linear.weight")
    if kind == "OneHiddenLayerMLP":
        return ("model.network.0.weight",)
    raise ValueError(f"Unsupported stored probe kind: {kind}")
