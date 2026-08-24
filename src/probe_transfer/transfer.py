from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.tracking import Tracker
from probe_transfer.activations import load_activation_split
from probe_transfer.artifacts import save_probe_bundle, write_jsonl
from probe_transfer.evaluation import (
    binary_metrics,
    fixed_operating_point_metrics,
    paired_auroc_gap_interval,
    prediction_rows,
    select_balanced_accuracy_threshold,
    select_fpr_thresholds,
)
from probe_transfer.training import ProbeSelection, families_for_depth, fit_probe_family

Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def run_transfer(
    output_dir: Path, config: dict[str, Any], tracker: Tracker
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    models = list(config["models"])
    probes_config = config["probes"]
    primary_depth = config["activations"]["primary_depth"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    score_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    bundles: dict[tuple[int, str], dict[str, Any]] = {}
    details: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}

    for data_seed in config["data_seeds"]:
        for depth in config["activations"]["normalized_depths"]:
            layer = _layer_key(depth)
            datasets = {
                model: _load_model_splits(output_dir, model, data_seed, layer) for model in models
            }
            _assert_aligned(datasets)
            families = families_for_depth(probes_config, depth, primary_depth)
            selected: dict[tuple[str, str], ProbeSelection] = {}

            for source in models:
                train_x, _, train_y = datasets[source]["train"]
                validation_x, _, validation_y = datasets[source]["validation"]
                for family in families:
                    choice = fit_probe_family(
                        family,
                        train_x,
                        train_y,
                        validation_x,
                        validation_y,
                        probes_config,
                        device=device,
                    )
                    selected[source, family] = choice
                    name = f"{layer}.{family}"
                    bundles.setdefault((data_seed, source), {})[name] = choice.probe
                    details.setdefault((data_seed, source), {})[name] = {
                        "data_seed": data_seed,
                        "depth": depth,
                        "family": family,
                        "source_model": source,
                        **choice.parameters,
                    }
                    tracker.metrics(
                        {
                            f"training/{source}/seed_{data_seed}/{name}/validation_auroc": (
                                choice.probe.validation_auroc
                            )
                        }
                    )

            _evaluate_layer(
                selected,
                datasets,
                models,
                config,
                data_seed,
                depth,
                metrics,
                predictions,
                score_cache,
                tracker,
            )

    checksums = _save_outputs(output_dir, bundles, details, metrics, predictions)
    gaps = _gap_rows(config, metrics, score_cache)
    checksums["results/transfer_gaps.jsonl"] = write_jsonl(
        output_dir / "results" / "transfer_gaps.jsonl", gaps
    )
    return gaps, checksums


def _load_model_splits(
    output_dir: Path, model: str, data_seed: int, layer: str
) -> dict[str, Split]:
    directory = output_dir / "activations" / model
    paths = {
        "train": directory / f"seed_{data_seed}_train.safetensors",
        "validation": directory / f"seed_{data_seed}_validation.safetensors",
        "test": directory / "test.safetensors",
    }
    loaded = {}
    for split, path in paths.items():
        activations, row_ids, labels = load_activation_split(path, layer)
        loaded[split] = (activations.numpy(), row_ids.numpy(), labels.numpy())
    return loaded


def _assert_aligned(datasets: dict[str, dict[str, Split]]) -> None:
    models = list(datasets)
    reference = datasets[models[0]]
    for model in models[1:]:
        for split in reference:
            _, reference_ids, reference_labels = reference[split]
            _, row_ids, labels = datasets[model][split]
            if not np.array_equal(reference_ids, row_ids) or not np.array_equal(
                reference_labels, labels
            ):
                raise ValueError(f"Models are not aligned on {split} rows.")


def _evaluate_layer(
    selected: dict[tuple[str, str], ProbeSelection],
    datasets: dict[str, dict[str, Split]],
    models: list[str],
    config: dict[str, Any],
    data_seed: int,
    depth: float,
    metrics: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    score_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]],
    tracker: Tracker,
) -> None:
    target_fprs = config["evaluation"]["operating_fprs"]
    for (source, family), choice in selected.items():
        validation_x, _, validation_y = datasets[source]["validation"]
        validation_scores = choice.probe.scores(validation_x)
        threshold = select_balanced_accuracy_threshold(validation_y, validation_scores)
        fpr_thresholds = select_fpr_thresholds(validation_y, validation_scores, target_fprs)

        for target in models:
            test_x, row_ids, labels = datasets[target]["test"]
            scores = choice.probe.scores(test_x)
            values = binary_metrics(labels, scores, threshold=threshold, target_fprs=target_fprs)
            values.update(fixed_operating_point_metrics(labels, scores, fpr_thresholds))
            context = {
                "data_seed": data_seed,
                "depth": depth,
                "probe_family": family,
                "source_model": source,
                "evaluation_model": target,
            }
            record = {**context, **values, **choice.parameters}
            metrics.append(record)
            for row in prediction_rows(row_ids.tolist(), labels, scores, threshold):
                predictions.append({**context, **row})
            score_cache[data_seed, depth, family, source, target] = (labels, scores)
            tracker.metrics(
                {
                    f"evaluation/{source}_to_{target}/seed_{data_seed}/"
                    f"{_layer_key(depth)}.{family}/{name}": float(value)
                    for name, value in values.items()
                }
            )


def _gap_rows(
    config: dict[str, Any],
    metrics: list[dict[str, Any]],
    scores: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            row["data_seed"],
            row["depth"],
            row["probe_family"],
            row["source_model"],
            row["evaluation_model"],
        ): row
        for row in metrics
    }
    models = list(config["models"])
    evaluation = config["evaluation"]
    gaps = []
    for key, (labels, transfer_scores) in scores.items():
        data_seed, depth, family, source, target = key
        if source == target:
            continue
        oracle_key = (data_seed, depth, family, target, target)
        oracle_scores = scores[oracle_key][1]
        gap, lower, upper = paired_auroc_gap_interval(
            labels,
            oracle_scores,
            transfer_scores,
            samples=evaluation["bootstrap_samples"],
            confidence=evaluation["confidence_level"],
            seed=_bootstrap_seed(config["seed"], key, models),
        )
        source_auroc = lookup[data_seed, depth, family, source, source]["auroc"]
        target_auroc = lookup[oracle_key]["auroc"]
        failed = (
            source_auroc >= evaluation["oracle_gate"]
            and target_auroc >= evaluation["oracle_gate"]
            and gap >= evaluation["minimum_gap"]
            and lower > 0
        )
        gaps.append(
            {
                "data_seed": data_seed,
                "depth": depth,
                "probe_family": family,
                "source_model": source,
                "target_model": target,
                "pair_group": _pair_group(config, source, target),
                "source_auroc": source_auroc,
                "target_oracle_auroc": target_auroc,
                "transfer_auroc": lookup[key]["auroc"],
                "auroc_gap": gap,
                "ci_lower": lower,
                "ci_upper": upper,
                "transfer_failed": failed,
            }
        )
    return gaps


def _pair_group(config: dict[str, Any], source: str, target: str) -> str:
    groups = config["evaluation"].get("pair_groups")
    if groups is None:
        return "primary"
    for name, pairs in groups.items():
        if [source, target] in pairs:
            return str(name)
    raise ValueError(f"Transfer pair is not assigned to an evaluation group: {source} -> {target}")


def _save_outputs(
    output_dir: Path,
    bundles: dict[tuple[int, str], dict[str, Any]],
    details: dict[tuple[int, str], dict[str, dict[str, Any]]],
    metrics: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, str]:
    checksums = {
        "results/metrics.jsonl": write_jsonl(output_dir / "results" / "metrics.jsonl", metrics),
        "results/predictions.jsonl": write_jsonl(
            output_dir / "results" / "predictions.jsonl", predictions
        ),
    }
    for (seed, model), probes in bundles.items():
        relative = f"probes/seed_{seed}/{model}.safetensors"
        checksums[relative] = save_probe_bundle(output_dir / relative, probes, details[seed, model])
    return checksums


def _layer_key(depth: float) -> str:
    return f"layer_{round(depth * 100)}"


def _bootstrap_seed(base: int, key: tuple[Any, ...], models: list[str]) -> int:
    data_seed, depth, family, source, target = key
    families = ["linear", "cp_degree_2", "mlp"]
    return (
        base
        + int(data_seed) * 10_000
        + round(float(depth) * 100) * 100
        + families.index(str(family)) * 10
        + models.index(str(source)) * 2
        + models.index(str(target))
    )
