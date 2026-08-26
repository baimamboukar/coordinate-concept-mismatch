import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from probe_transfer.probes.models import LinearProbe, NeuralProbe

Probe = LinearProbe | NeuralProbe


def save_probe_bundle(
    path: str | Path,
    probes: Mapping[str, Probe],
    details: Mapping[str, dict[str, Any]],
) -> str:
    if set(probes) != set(details):
        raise ValueError("Probe tensors and details must use the same names.")

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    for name in sorted(probes):
        probe = probes[name]
        prefix = name.replace("/", ".")
        tensors[f"{prefix}.preprocessor.mean"] = torch.from_numpy(probe.preprocessor.mean).float()
        tensors[f"{prefix}.preprocessor.scale"] = torch.tensor(
            probe.preprocessor.scale, dtype=torch.float32
        )
        record = dict(details[name])
        record["validation_auroc"] = probe.validation_auroc

        if isinstance(probe, LinearProbe):
            tensors[f"{prefix}.coefficient"] = torch.from_numpy(probe.estimator.coef_).float()
            tensors[f"{prefix}.intercept"] = torch.from_numpy(probe.estimator.intercept_).float()
            record.update({"kind": "linear", "c": probe.c})
        else:
            for parameter, value in probe.model.state_dict().items():
                tensors[f"{prefix}.model.{parameter}"] = value.detach().cpu()
            record.update({"kind": type(probe.model).__name__, "epochs": probe.epochs})
        metadata[name] = record

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        output,
        metadata={"probes": json.dumps(metadata, sort_keys=True)},
    )
    return sha256_file(output)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    return sha256_file(output)


def write_json(path: str | Path, value: Any) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return sha256_file(output)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
