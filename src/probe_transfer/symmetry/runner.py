import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from core.constants import ACTIVATION_ROWS_ENV, BASELINE_ARTIFACT_ENV, EXPERIMENT_OUTPUT_ENV
from core.tracking import Tracker
from probe_transfer.artifacts import write_json, write_jsonl
from probe_transfer.atomic import publish_directories
from probe_transfer.data import load_prepared_rows
from probe_transfer.extraction.runtime import validate_cuda_runtime, validate_free_disk
from probe_transfer.symmetry.alignment import estimate_transformation_maps
from probe_transfer.symmetry.coordinates import CoordinateTransform
from probe_transfer.symmetry.evaluation import evaluate_transformations
from probe_transfer.symmetry.gate import run_function_gates
from probe_transfer.symmetry.protocol import estimated_alignment_enabled
from probe_transfer.symmetry.scales import seeded_positive_diagonal
from probe_transfer.symmetry.transforms import (
    seeded_gqa_head_permutation,
    seeded_permutation,
)


def run_symmetry_experiment(config: dict[str, Any], tracker: Tracker) -> None:
    baseline = _required_directory(BASELINE_ARTIFACT_ENV)
    prepared = _required_directory(ACTIVATION_ROWS_ENV)
    output = _required_directory(EXPERIMENT_OUTPUT_ENV, create=True)
    for name in ("probes", "results"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output / name}")

    symmetry = config["symmetry"]
    runtime = validate_cuda_runtime(config["execution"], symmetry["gate_dtype"])
    free_disk = validate_free_disk(output, config["execution"]["minimum_disk_free_gb"])
    tracker.report(
        "Runtime",
        f"{runtime['gpu']} with {runtime['memory_gb']:.1f} GiB; {free_disk:.1f} GiB free.",
    )
    transformations = _seeded_transformations(symmetry)
    rows = load_prepared_rows(
        prepared / "test.jsonl", config["materials"]["expected_test_rows"], require_balanced=False
    )
    with TemporaryDirectory(prefix=".symmetry-", dir=output) as temporary:
        staging = Path(temporary)
        prototype = next(iter(transformations.values()))
        write_json(
            staging / "results" / prototype.map_filename,
            {str(seed): item.values.tolist() for seed, item in transformations.items()},
        )
        smoke_rows = symmetry.get("smoke_gate_rows", 0)
        if smoke_rows:
            smoke = run_function_gates(config, rows[:smoke_rows], transformations)
            write_jsonl(staging / "results" / "function_gate_smoke.jsonl", smoke)
            _require_gate_pass(smoke, "fail-fast", output)

        gates = run_function_gates(config, rows, transformations)
        write_jsonl(staging / "results" / "function_gates.jsonl", gates)
        _require_gate_pass(gates, "full-test", output)

        maps, diagnostics = estimate_transformation_maps(baseline, config, transformations)
        if estimated_alignment_enabled(config):
            write_jsonl(
                staging / "results" / "alignment_diagnostics.jsonl",
                diagnostics,
            )
        recoveries, _ = evaluate_transformations(
            baseline,
            staging,
            config,
            transformations,
            estimated_maps=maps,
        )
        _validate_outputs(staging, config)
        primary = [row for row in recoveries if row["depth"] == symmetry["primary_depth"]]
        exact = sum(row["exact_recovery"] for row in primary)
        estimated = sum(row.get("estimated_recovery", False) for row in primary)
        tracker.metrics(
            {
                "symmetry/primary_comparisons": float(len(primary)),
                "symmetry/coordinate_failure_fraction": sum(
                    bool(row["coordinate_failure"]) for row in primary
                )
                / len(primary),
                "symmetry/mean_raw_auroc_gap": sum(float(row["raw_auroc_gap"]) for row in primary)
                / len(primary),
                "symmetry/exact_recovery_fraction": exact / len(primary),
                "symmetry/estimated_recovery_fraction": estimated / len(primary),
                "symmetry/maximum_logit_error": max(
                    float(gate["maximum_logit_error"]) for gate in gates
                ),
                "symmetry/maximum_activation_relative_error": max(
                    float(gate["maximum_activation_relative_error"]) for gate in gates
                ),
            }
        )
        tracker.report(
            "Summary",
            f"All {len(gates)} function-preservation gates passed. Analytic transport recovered "
            f"{exact}/{len(primary)} primary comparisons; activation-estimated alignment "
            f"recovered {estimated}/{len(primary)}.",
        )
        publish_directories(staging, output, ("probes", "results"))


def _require_gate_pass(gates: list[dict[str, Any]], phase: str, output: Path) -> None:
    failed = [gate for gate in gates if not gate["passed"]]
    if failed:
        write_jsonl(output / "diagnostics" / f"{phase}_function_gates.jsonl", gates)
        details = "; ".join(
            f"seed={gate['transformation_seed']} logits={gate['maximum_logit_error']:.3e} "
            f"activations={gate['maximum_activation_relative_error']:.3e} "
            f"agreement={gate['next_token_agreement']:.6f}"
            for gate in failed
        )
        raise RuntimeError(f"{len(failed)} {phase} function-preservation gates failed: {details}")


def _validate_outputs(output: Path, config: dict[str, Any]) -> None:
    expected = config["expected_outputs"]
    paths = {
        "metrics_rows": output / "results" / "metrics.jsonl",
        "prediction_rows": output / "results" / "predictions.jsonl",
        "recovery_rows": output / "results" / "recovery.jsonl",
        "function_gate_rows": output / "results" / "function_gates.jsonl",
    }
    if "function_smoke_gate_rows" in expected:
        paths["function_smoke_gate_rows"] = output / "results" / "function_gate_smoke.jsonl"
    if "alignment_diagnostic_rows" in expected:
        paths["alignment_diagnostic_rows"] = output / "results" / "alignment_diagnostics.jsonl"
    for name, path in paths.items():
        with path.open("rb") as handle:
            actual = sum(1 for _ in handle)
        if actual != expected[name]:
            raise ValueError(f"Expected {expected[name]} rows in {path}, found {actual}.")
    label = (
        "scale"
        if config["symmetry"]["transformation"] == "mlp_positive_diagonal"
        else "permutation"
    )
    bundles = list((output / "probes").glob(f"{label}_*/seed_*/*.safetensors"))
    if len(bundles) != expected["probe_bundles"]:
        raise ValueError("Unexpected number of transported probe bundles.")
    for path in (output / "results").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if any(
                isinstance(value, float) and not math.isfinite(value)
                for value in json.loads(line).values()
            ):
                raise ValueError(f"Non-finite value in {path}.")


def _required_directory(name: str, *, create: bool = False) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required.")
    path = Path(value).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return path


def _seeded_transformations(symmetry: dict[str, Any]) -> dict[int, CoordinateTransform]:
    seeds = symmetry["transformation_seeds"]
    if symmetry["transformation"] == "mlp_positive_diagonal":
        minimum, maximum = symmetry["scale_range"]
        return {
            seed: CoordinateTransform(
                "positive_diagonal",
                seeded_positive_diagonal(symmetry["width"], seed, minimum, maximum),
            )
            for seed in seeds
        }
    if symmetry["transformation"] != "attention_head_permutation":
        return {
            seed: CoordinateTransform("permutation", seeded_permutation(symmetry["width"], seed))
            for seed in seeds
        }
    layout = symmetry["attention_layout"]
    return {
        seed: CoordinateTransform(
            "permutation",
            seeded_gqa_head_permutation(
                layout["query_heads"],
                layout["key_value_heads"],
                layout["head_dim"],
                seed,
            ),
        )
        for seed in seeds
    }
