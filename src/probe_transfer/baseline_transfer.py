import os
from pathlib import Path
from typing import Any

from safetensors import safe_open

from core.tracking import Tracker
from probe_transfer.transfer import run_transfer


def run_staged_transfer(config: dict[str, Any], tracker: Tracker) -> None:
    output_dir = _output_directory()
    _validate_activations(output_dir, config)
    for name in ("probes", "results"):
        path = output_dir / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    gaps, _ = run_transfer(output_dir, config, tracker)
    primary = [row for row in gaps if row["pair_group"] == "primary"]
    failures = sum(bool(row["transfer_failed"]) for row in primary)
    tracker.report(
        "Summary",
        f"Completed {len(gaps)} directed probe-transfer comparisons across all layers, "
        f"probe families, and data seeds. The primary Llama/Qwen group contained "
        f"{len(primary)} comparisons, of which {failures} met the direction-level "
        "prespecified failure rule.",
    )


def _output_directory() -> Path:
    value = os.getenv("ACTIVATION_STAGING_DIR")
    if not value:
        raise RuntimeError("ACTIVATION_STAGING_DIR is required for staged transfer.")
    return Path(value).expanduser().resolve()


def _validate_activations(output_dir: Path, config: dict[str, Any]) -> None:
    sampling = config["sampling"]
    expected = {
        "test": sampling["test_size"],
        **{
            f"seed_{seed}_{split}": sampling[f"{split}_size"]
            for seed in config["data_seeds"]
            for split in ("train", "validation")
        },
    }
    layer_keys = [
        f"layer_{round(depth * 100)}" for depth in config["activations"]["normalized_depths"]
    ]

    missing = []
    for model_name, model in config["models"].items():
        model_dir = output_dir / "activations" / model_name
        completion = model_dir / "completion.json"
        if not completion.is_file():
            missing.append(str(completion))
        for split, rows in expected.items():
            path = model_dir / f"{split}.safetensors"
            if not path.is_file():
                missing.append(str(path))
                continue
            with safe_open(path, framework="pt", device="cpu") as saved:
                required = {"row_ids", "labels", *layer_keys}
                absent = required - set(saved.keys())
                if absent:
                    raise ValueError(f"{path} is missing tensors: {sorted(absent)}")
                if saved.get_slice("row_ids").get_shape() != [rows] or saved.get_slice(
                    "labels"
                ).get_shape() != [rows]:
                    raise ValueError(f"{path} does not contain the expected {rows} rows.")
                for key in layer_keys:
                    shape = saved.get_slice(key).get_shape()
                    if shape != [rows, model["hidden_size"]]:
                        raise ValueError(f"Unexpected activation shape for {path}:{key}: {shape}")

    if missing:
        raise FileNotFoundError(f"Missing staged activation files: {missing}")
