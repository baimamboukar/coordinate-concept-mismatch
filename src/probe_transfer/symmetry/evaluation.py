import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from probe_transfer.alignment.methods import AlignmentMap
from probe_transfer.artifacts import sha256_file, write_jsonl
from probe_transfer.extraction.activations import load_activation_split
from probe_transfer.probes.evaluation import (
    binary_metrics,
    fixed_operating_point_metrics,
    prediction_rows,
)
from probe_transfer.probes.training import families_for_depth
from probe_transfer.probes.transport import (
    StoredProbe,
    load_probe_bundle,
    save_probe_bundle,
)
from probe_transfer.symmetry.protocol import estimated_alignment_enabled, selected_models
from probe_transfer.symmetry.recovery import recovery_record
from probe_transfer.symmetry.transforms import inverse_permutation


def evaluate_permutations(
    baseline_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    permutations: dict[int, torch.Tensor],
    estimated_maps: dict[tuple[int, str, float, int], AlignmentMap] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    baseline_metrics = _reference_metrics(baseline_dir / "results" / "metrics.jsonl")
    metrics: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}
    prediction_path = output_dir / "results" / "predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_index = 0
    with prediction_path.open("w") as prediction_file:
        for data_seed in config["data_seeds"]:
            for model_name in selected_models(config):
                probes = _select_probes(
                    load_probe_bundle(
                        baseline_dir / "probes" / f"seed_{data_seed}" / f"{model_name}.safetensors"
                    ),
                    config,
                )
                _save_transported_probes(
                    output_dir,
                    probes,
                    model_name,
                    data_seed,
                    permutations,
                    checksums,
                )
                for probe_name, probe in sorted(probes.items()):
                    depth, family = _parse_probe_name(probe_name)
                    activations, row_ids, labels = load_activation_split(
                        baseline_dir / "activations" / model_name / "test.safetensors",
                        f"layer_{round(depth * 100)}",
                    )
                    values = activations.numpy()
                    label_values = labels.numpy()
                    reference = baseline_metrics[data_seed, depth, family, model_name]
                    reference_scores = probe.scores(values)
                    _assert_reference(reference_scores, label_values, reference, config)
                    thresholds = _operating_thresholds(reference, config)
                    context = {
                        "data_seed": data_seed,
                        "model": model_name,
                        "depth": depth,
                        "probe_family": family,
                    }
                    _record_condition(
                        metrics,
                        prediction_file,
                        context,
                        "reference",
                        None,
                        row_ids.numpy(),
                        label_values,
                        reference_scores,
                        reference["threshold"],
                        thresholds,
                        config,
                    )

                    identity = torch.arange(values.shape[1])
                    identity_scores = probe.transport(identity).scores(values[:, identity])
                    _assert_score_match(reference_scores, identity_scores, config)
                    _record_condition(
                        metrics,
                        prediction_file,
                        context,
                        "identity_control",
                        None,
                        row_ids.numpy(),
                        label_values,
                        identity_scores,
                        reference["threshold"],
                        thresholds,
                        config,
                    )

                    for permutation_seed, permutation in permutations.items():
                        permuted_values = values[:, permutation]
                        raw_scores = probe.scores(permuted_values)
                        transported_scores = probe.transport(permutation).scores(permuted_values)
                        inverse_scores = probe.transport(inverse_permutation(permutation)).scores(
                            permuted_values
                        )
                        conditions = [
                            ("raw_permuted", raw_scores),
                            ("exact_transport", transported_scores),
                            ("inverse_transport", inverse_scores),
                        ]
                        estimated_scores = None
                        if estimated_alignment_enabled(config):
                            if estimated_maps is None:
                                raise ValueError("Estimated symmetry alignment maps are required.")
                            key = (data_seed, model_name, depth, permutation_seed)
                            estimated_scores = probe.scores(
                                estimated_maps[key].transform(permuted_values)
                            )
                            conditions.insert(2, ("estimated_alignment", estimated_scores))
                        for condition, scores in conditions:
                            _record_condition(
                                metrics,
                                prediction_file,
                                context,
                                condition,
                                permutation_seed,
                                row_ids.numpy(),
                                label_values,
                                scores,
                                reference["threshold"],
                                thresholds,
                                config,
                            )
                        recoveries.append(
                            recovery_record(
                                context,
                                permutation_seed,
                                label_values,
                                reference_scores,
                                raw_scores,
                                transported_scores,
                                inverse_scores,
                                estimated_scores,
                                config,
                                comparison_index,
                            )
                        )
                        comparison_index += 1

    checksums["results/predictions.jsonl"] = sha256_file(prediction_path)
    checksums["results/metrics.jsonl"] = write_jsonl(
        output_dir / "results" / "metrics.jsonl", metrics
    )
    checksums["results/recovery.jsonl"] = write_jsonl(
        output_dir / "results" / "recovery.jsonl", recoveries
    )
    return recoveries, checksums


def _record_condition(
    metrics: list[dict[str, Any]],
    prediction_file: TextIO,
    context: dict[str, Any],
    condition: str,
    permutation_seed: int | None,
    row_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    operating_thresholds: dict[float, float],
    config: dict[str, Any],
) -> None:
    values = binary_metrics(
        labels,
        scores,
        threshold=float(threshold),
        target_fprs=config["evaluation"]["operating_fprs"],
    )
    values.update(fixed_operating_point_metrics(labels, scores, operating_thresholds))
    record = {
        **context,
        "condition": condition,
        "permutation_seed": permutation_seed,
        **values,
    }
    metrics.append(record)
    prediction_file.writelines(
        json.dumps(
            {
                **context,
                "condition": condition,
                "permutation_seed": permutation_seed,
                **row,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in prediction_rows(row_ids.tolist(), labels, scores, float(threshold))
    )


def _save_transported_probes(
    output_dir: Path,
    probes: dict[str, StoredProbe],
    model: str,
    data_seed: int,
    permutations: dict[int, torch.Tensor],
    checksums: dict[str, str],
) -> None:
    for permutation_seed, permutation in permutations.items():
        transported = {name: probe.transport(permutation) for name, probe in probes.items()}
        relative = f"probes/permutation_{permutation_seed}/seed_{data_seed}/{model}.safetensors"
        checksums[relative] = save_probe_bundle(
            output_dir / relative,
            transported,
            metadata_updates={"permutation_seed": permutation_seed},
        )


def _reference_metrics(path: Path) -> dict[tuple[int, float, str, str], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {
        (row["data_seed"], row["depth"], row["probe_family"], row["source_model"]): row
        for row in rows
        if row["source_model"] == row["evaluation_model"]
    }


def _operating_thresholds(reference: dict[str, Any], config: dict[str, Any]) -> dict[float, float]:
    return {
        float(target): float(reference[f"source_threshold_{target * 100:g}pct"])
        for target in config["evaluation"]["operating_fprs"]
    }


def _assert_reference(
    scores: np.ndarray,
    labels: np.ndarray,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> None:
    actual = float(roc_auc_score(labels, scores))
    if abs(actual - float(reference["auroc"])) > config["evaluation"]["reference_auroc_atol"]:
        raise ValueError("Stored probe scores do not reproduce the baseline reference AUROC.")


def _assert_score_match(reference: np.ndarray, actual: np.ndarray, config: dict[str, Any]) -> None:
    evaluation = config["evaluation"]
    if not np.allclose(
        actual,
        reference,
        atol=evaluation["score_atol"],
        rtol=evaluation["score_rtol"],
    ):
        raise ValueError("Identity probe transport changed stored probe scores.")


def _parse_probe_name(name: str) -> tuple[float, str]:
    layer, family = name.split(".", maxsplit=1)
    return int(layer.removeprefix("layer_")) / 100, family


def _select_probes(
    probes: dict[str, StoredProbe], config: dict[str, Any]
) -> dict[str, StoredProbe]:
    symmetry = config["symmetry"]
    expected = {
        f"layer_{round(depth * 100)}.{family}"
        for depth in symmetry["probed_depths"]
        for family in families_for_depth(config["probes"], depth, symmetry["primary_depth"])
    }
    missing = expected.difference(probes)
    if missing:
        raise ValueError(f"Probe bundle is missing configured probes: {sorted(missing)}")
    return {name: probes[name] for name in sorted(expected)}
