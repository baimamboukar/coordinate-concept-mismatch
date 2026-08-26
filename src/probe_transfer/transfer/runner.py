import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from safetensors import safe_open

from core.constants import ACTIVATION_STAGING_ENV
from core.tracking import Tracker
from probe_transfer.atomic import publish_directories
from probe_transfer.transfer.evaluation import run_transfer


def run_staged_transfer(config: dict[str, Any], tracker: Tracker) -> None:
    output_dir = _output_directory(config)
    _validate_activations(output_dir, config)
    for name in ("probes", "results"):
        path = output_dir / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    with TemporaryDirectory(prefix=".transfer-", dir=output_dir) as temporary:
        staging = Path(temporary)
        (staging / "activations").symlink_to(output_dir / "activations", target_is_directory=True)
        gaps, _ = run_transfer(staging, config, tracker)
        _validate_outputs(staging, config)
        publish_directories(staging, output_dir, ("probes", "results"))
    primary = [row for row in gaps if row["pair_group"] == "primary"]
    failures = sum(bool(row["transfer_failed"]) for row in primary)
    tracker.report(
        "Summary",
        f"Completed {len(gaps)} directed probe-transfer comparisons across all layers, "
        f"probe families, and data seeds. The configured primary group contained "
        f"{len(primary)} comparisons, of which {failures} met the direction-level "
        "prespecified failure rule.",
    )


def _output_directory(config: dict[str, Any]) -> Path:
    value = os.getenv(ACTIVATION_STAGING_ENV) or config.get("extraction", {}).get("staging_dir")
    if not value:
        raise RuntimeError(f"{ACTIVATION_STAGING_ENV} is required for staged transfer.")
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
        for split, rows in expected.items():
            path = model_dir / f"{split}.safetensors"
            if not path.is_file():
                missing.append(str(path))
                continue
            with safe_open(path, framework="pt", device="cpu") as saved:
                _validate_metadata(saved.metadata(), model, split, rows, config)
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


def _validate_metadata(
    metadata: dict[str, str] | None,
    model: dict[str, Any],
    split: str,
    rows: int,
    config: dict[str, Any],
) -> None:
    if metadata is None:
        raise ValueError("Activation files require lineage metadata.")
    expected = {
        "model": model["id"],
        "model_revision": model["revision"],
        "dataset": config["dataset"]["id"],
        "dataset_revision": config["dataset"]["revision"],
        "split": split,
        "rows": str(rows),
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(f"Activation lineage metadata changed for {model['id']}/{split}.")


def _validate_outputs(output_dir: Path, config: dict[str, Any]) -> None:
    expected = config["expected_outputs"]
    contracts = {
        "metrics_rows": (
            output_dir / "results" / "metrics.jsonl",
            {"data_seed", "depth", "probe_family", "source_model", "evaluation_model", "auroc"},
        ),
        "prediction_rows": (
            output_dir / "results" / "predictions.jsonl",
            {
                "data_seed",
                "depth",
                "probe_family",
                "source_model",
                "evaluation_model",
                "row_id",
                "label",
                "score",
                "probability",
                "prediction",
            },
        ),
        "transfer_gap_rows": (
            output_dir / "results" / "transfer_gaps.jsonl",
            {
                "data_seed",
                "depth",
                "probe_family",
                "source_model",
                "target_model",
                "pair_group",
                "auroc_gap",
                "ci_lower",
                "ci_upper",
                "transfer_failed",
            },
        ),
    }
    for name, (path, fields) in contracts.items():
        _validate_jsonl(path, expected[name], fields)

    actual_bundles = {
        path.relative_to(output_dir).as_posix()
        for path in (output_dir / "probes").glob("seed_*/*.safetensors")
    }
    expected_bundles = {
        f"probes/seed_{seed}/{model}.safetensors"
        for seed in config["data_seeds"]
        for model in config["models"]
    }
    if actual_bundles != expected_bundles or len(actual_bundles) != expected["probe_bundles"]:
        raise ValueError("Probe bundle output contract changed.")


def _validate_jsonl(path: Path, expected_rows: int, required: set[str]) -> None:
    rows = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if not required.issubset(row):
                raise ValueError(f"Missing required fields in {path}.")
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise ValueError(f"Non-finite value in {path}.")
            rows += 1
    if rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, found {rows}.")
