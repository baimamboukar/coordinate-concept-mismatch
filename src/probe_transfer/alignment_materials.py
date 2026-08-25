import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from probe_transfer.activations import load_activation_split
from probe_transfer.alignment_quotient import build_quotient_basis
from probe_transfer.probe_transport import StoredProbe, load_probe_bundle


def paired_split(
    root: Path, source: str, target: str, split: str, layer: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loaded = []
    for model in (source, target):
        values, row_ids, labels = load_activation_split(
            root / "activations" / model / f"{split}.safetensors", layer
        )
        loaded.append((values.numpy(), row_ids.numpy(), labels.numpy()))
    if not np.array_equal(loaded[0][1], loaded[1][1]) or not np.array_equal(
        loaded[0][2], loaded[1][2]
    ):
        raise ValueError(f"Checkpoint activations are not paired for {split}/{layer}.")
    return loaded[0][0], loaded[1][0], loaded[0][1], loaded[0][2]


def load_bundles(
    root: Path, config: dict[str, Any]
) -> dict[tuple[int, str], dict[str, StoredProbe]]:
    return {
        (seed, model): load_probe_bundle(root / "probes" / f"seed_{seed}" / f"{model}.safetensors")
        for seed in config["data_seeds"]
        for model in config["models"]
    }


def quotient_basis(
    bundles: dict[tuple[int, str], dict[str, StoredProbe]],
    seeds: list[int],
    source: str,
    layer: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float | int]]:
    probes = [bundles[seed, source][f"{layer}.linear"] for seed in seeds]
    return build_quotient_basis(probes, config["quotient_svd_relative_threshold"])


def load_baseline_metrics(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {
        (
            row["data_seed"],
            row["depth"],
            row["probe_family"],
            row["source_model"],
            row["evaluation_model"],
        ): row
        for row in rows
    }


def references(
    rows: dict[tuple[Any, ...], dict[str, Any]],
    seed: int,
    depth: float,
    family: str,
    source: str,
    target: str,
) -> dict[str, dict[str, Any]]:
    return {
        "source": rows[seed, depth, family, source, source],
        "raw": rows[seed, depth, family, source, target],
        "target": rows[seed, depth, family, target, target],
    }


def assert_references(
    labels: np.ndarray,
    source: np.ndarray,
    raw: np.ndarray,
    target: np.ndarray,
    expected: dict[str, dict[str, Any]],
    tolerance: float,
) -> None:
    for name, scores in (("source", source), ("raw", raw), ("target", target)):
        actual = float(roc_auc_score(labels, scores))
        if abs(actual - float(expected[name]["auroc"])) > tolerance:
            raise ValueError(f"Stored {name} probe does not reproduce its baseline AUROC.")


def operating_thresholds(reference: dict[str, Any], target_fprs: list[float]) -> dict[float, float]:
    return {
        float(target): float(reference[f"source_threshold_{target * 100:g}pct"])
        for target in target_fprs
    }


def families(depth: float, primary_depth: float) -> list[str]:
    return ["linear", "cp_degree_2", "mlp"] if depth == primary_depth else ["linear"]


def layer_key(depth: float) -> str:
    return f"layer_{round(depth * 100)}"


def directions(models: list[str]) -> list[tuple[str, str]]:
    if len(models) != 2:
        raise ValueError("Checkpoint alignment requires exactly two models.")
    return [(models[0], models[1]), (models[1], models[0])]


def direction_groups(config: dict[str, Any]) -> list[tuple[str, str, str]]:
    groups = config.get("evaluation", {}).get("pair_groups")
    if groups is None:
        return [(*pair, "primary") for pair in directions(list(config["models"]))]

    models = set(config["models"])
    grouped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for group, pairs in groups.items():
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Invalid model pair in {group}: {pair}")
            source, target = pair
            if source not in models or target not in models or source == target:
                raise ValueError(f"Invalid alignment direction in {group}: {pair}")
            if (source, target) in seen:
                raise ValueError(f"Duplicate alignment direction: {source} -> {target}")
            seen.add((source, target))
            grouped.append((source, target, str(group)))
    if not grouped:
        raise ValueError("At least one alignment direction is required.")
    return grouped


def resolve_device(configured: str) -> str:
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported alignment device: {configured}")
    if configured == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA alignment was requested but is unavailable.")
    return configured
