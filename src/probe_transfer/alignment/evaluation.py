import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from probe_transfer.alignment.cross_task import (
    add_improvement_retention,
    load_recovery_reference,
)
from probe_transfer.alignment.materials import (
    assert_references,
    direction_groups,
    families,
    layer_key,
    load_baseline_metrics,
    load_bundles,
    operating_thresholds,
    paired_split,
    quotient_basis,
    references,
    resolve_device,
)
from probe_transfer.alignment.methods import alignment_diagnostic, fit_ambient_alignments
from probe_transfer.alignment.quotient import (
    fit_quotient_alignment,
    quotient_scores,
)
from probe_transfer.alignment.recovery import alignment_recovery_record
from probe_transfer.artifacts import sha256_file, write_jsonl
from probe_transfer.probes.evaluation import (
    binary_metrics,
    fixed_operating_point_metrics,
    prediction_rows,
)


def evaluate_checkpoint_alignment(
    baseline_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    *,
    fit_root: Path | None = None,
    reference_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    baseline = load_baseline_metrics(baseline_dir / "results" / "metrics.jsonl")
    bundles = load_bundles(baseline_dir, config)
    alignment = config["alignment"]
    device = resolve_device(alignment["device"])
    metrics: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    prediction_path = output_dir / "results" / "predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_index = 0
    materials = config["materials"]
    fit_materials = config.get("fit_materials", materials)
    fit_root = fit_root or baseline_dir
    recovery_reference = load_recovery_reference(reference_path)
    with prediction_path.open("w") as prediction_file:
        for data_seed in config["data_seeds"]:
            for source, target, pair_group in direction_groups(config):
                for depth in alignment["depths"]:
                    layer = layer_key(depth)
                    train = paired_split(fit_root, source, target, f"seed_{data_seed}_train", layer)
                    validation = paired_split(
                        baseline_dir, source, target, f"seed_{data_seed}_validation", layer
                    )
                    test = paired_split(baseline_dir, source, target, "test", layer)
                    _assert_row_count(train, fit_materials["expected_train_rows"], "fit train")
                    _assert_row_count(
                        validation, materials["expected_validation_rows"], "validation"
                    )
                    _assert_row_count(test, materials["expected_test_rows"], "test")
                    prepared_probes = []
                    for family in families(config, depth):
                        source_probe = bundles[data_seed, source][f"{layer}.{family}"]
                        target_probe = bundles[data_seed, target][f"{layer}.{family}"]
                        source_scores = source_probe.scores(test[0])
                        raw_scores = source_probe.scores(test[1])
                        oracle_scores = target_probe.scores(test[1])
                        expected = references(baseline, data_seed, depth, family, source, target)
                        assert_references(
                            test[3],
                            source_scores,
                            raw_scores,
                            oracle_scores,
                            expected,
                            config["evaluation"]["reference_auroc_atol"],
                        )
                        prepared_probes.append(
                            (family, source_probe, raw_scores, oracle_scores, expected)
                        )

                    maps = fit_ambient_alignments(
                        train[0],
                        train[1],
                        relative_alpha=alignment["ridge_relative_alpha"],
                        shuffle_seed=alignment["shuffled_pairing_seed"] + comparison_index,
                        device=device,
                    )
                    selected_maps = {
                        name: maps[name]
                        for name in [*alignment["methods"], alignment["negative_control"]]
                        if name != "quotient_ridge"
                    }
                    aligned_test = {
                        name: fitted.transform(test[1]) for name, fitted in selected_maps.items()
                    }
                    diagnostics.extend(
                        _ambient_diagnostics(
                            selected_maps,
                            validation[0],
                            validation[1],
                            data_seed,
                            depth,
                            source,
                            target,
                            pair_group,
                        )
                    )

                    quotient_basis_values, quotient_metadata = quotient_basis(
                        bundles, config["data_seeds"], source, layer, alignment
                    )
                    quotient = fit_quotient_alignment(
                        train[0],
                        train[1],
                        quotient_basis_values,
                        relative_alpha=alignment["ridge_relative_alpha"],
                        device=device,
                    )
                    quotient_test = quotient.transform(test[1])
                    quotient_expected = validation[0] @ quotient_basis_values.T
                    diagnostics.append(
                        {
                            "data_seed": data_seed,
                            "depth": depth,
                            "source_model": source,
                            "target_model": target,
                            "pair_group": pair_group,
                            "method": "quotient_ridge",
                            **quotient_metadata,
                            **alignment_diagnostic(quotient, quotient_expected, validation[1]),
                        }
                    )

                    for (
                        family,
                        source_probe,
                        raw_scores,
                        oracle_scores,
                        expected,
                    ) in prepared_probes:
                        context = {
                            "data_seed": data_seed,
                            "depth": depth,
                            "probe_family": family,
                            "source_model": source,
                            "target_model": target,
                            "pair_group": pair_group,
                        }
                        _record(
                            metrics,
                            prediction_file,
                            context,
                            "raw_transfer",
                            test[2],
                            test[3],
                            raw_scores,
                            expected["source"],
                            config,
                        )
                        _record(
                            metrics,
                            prediction_file,
                            context,
                            "target_oracle",
                            test[2],
                            test[3],
                            oracle_scores,
                            expected["target"],
                            config,
                        )
                        method_scores = {
                            name: source_probe.scores(values)
                            for name, values in aligned_test.items()
                        }
                        if family == "linear":
                            method_scores["quotient_ridge"] = quotient_scores(
                                source_probe, quotient_test, quotient_basis_values
                            )
                        for method, scores in method_scores.items():
                            _record(
                                metrics,
                                prediction_file,
                                context,
                                method,
                                test[2],
                                test[3],
                                scores,
                                expected["source"],
                                config,
                            )
                            recovery = alignment_recovery_record(
                                {**context, "method": method},
                                test[3],
                                oracle_scores,
                                raw_scores,
                                scores,
                                float(expected["source"]["auroc"]),
                                config,
                                config["seed"] + comparison_index,
                            )
                            recoveries.append(
                                add_improvement_retention(
                                    recovery,
                                    recovery_reference,
                                    tolerance=config["evaluation"]["reference_auroc_atol"],
                                )
                            )
                            comparison_index += 1

    checksums = {
        "results/predictions.jsonl": sha256_file(prediction_path),
        "results/metrics.jsonl": write_jsonl(output_dir / "results" / "metrics.jsonl", metrics),
        "results/recovery.jsonl": write_jsonl(
            output_dir / "results" / "recovery.jsonl", recoveries
        ),
        "results/alignment_diagnostics.jsonl": write_jsonl(
            output_dir / "results" / "alignment_diagnostics.jsonl", diagnostics
        ),
    }
    return recoveries, diagnostics, checksums


def _assert_row_count(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: int,
    split: str,
) -> None:
    if any(len(item) != expected for item in values):
        raise ValueError(f"Expected {expected} paired {split} rows.")


def _record(
    metrics: list[dict[str, Any]],
    output: TextIO,
    context: dict[str, Any],
    condition: str,
    row_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> None:
    thresholds = operating_thresholds(reference, config["evaluation"]["operating_fprs"])
    values = binary_metrics(
        labels,
        scores,
        threshold=float(reference["threshold"]),
        target_fprs=config["evaluation"]["operating_fprs"],
    )
    values.update(fixed_operating_point_metrics(labels, scores, thresholds))
    metrics.append({**context, "condition": condition, **values})
    output.writelines(
        json.dumps(
            {**context, "condition": condition, **row},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in prediction_rows(row_ids.tolist(), labels, scores, float(reference["threshold"]))
    )


def _ambient_diagnostics(
    maps: dict[str, Any],
    source: np.ndarray,
    target: np.ndarray,
    seed: int,
    depth: float,
    source_model: str,
    target_model: str,
    pair_group: str,
) -> list[dict[str, Any]]:
    return [
        {
            "data_seed": seed,
            "depth": depth,
            "source_model": source_model,
            "target_model": target_model,
            "pair_group": pair_group,
            "method": name,
            **fitted.metadata,
            **alignment_diagnostic(fitted, source, target),
        }
        for name, fitted in maps.items()
    ]
