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
from probe_transfer.symmetry.alignment import estimate_permutation_maps
from probe_transfer.symmetry.evaluation import evaluate_permutations
from probe_transfer.symmetry.gate import run_function_gates
from probe_transfer.symmetry.protocol import estimated_alignment_enabled
from probe_transfer.symmetry.transforms import seeded_permutation


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
    permutations = {
        seed: seeded_permutation(symmetry["width"], seed) for seed in symmetry["permutation_seeds"]
    }
    rows = load_prepared_rows(
        prepared / "test.jsonl", config["materials"]["expected_test_rows"], require_balanced=False
    )
    with TemporaryDirectory(prefix=".symmetry-", dir=output) as temporary:
        staging = Path(temporary)
        write_json(
            staging / "results" / "permutations.json",
            {str(seed): permutation.tolist() for seed, permutation in permutations.items()},
        )
        smoke_rows = symmetry.get("smoke_gate_rows", 0)
        if smoke_rows:
            smoke = run_function_gates(config, rows[:smoke_rows], permutations)
            write_jsonl(staging / "results" / "function_gate_smoke.jsonl", smoke)
            _require_gate_pass(smoke, "fail-fast", output)

        gates = run_function_gates(config, rows, permutations)
        write_jsonl(staging / "results" / "function_gates.jsonl", gates)
        _require_gate_pass(gates, "full-test", output)

        maps, diagnostics = estimate_permutation_maps(baseline, config, permutations)
        if estimated_alignment_enabled(config):
            write_jsonl(
                staging / "results" / "alignment_diagnostics.jsonl",
                diagnostics,
            )
        recoveries, _ = evaluate_permutations(
            baseline,
            staging,
            config,
            permutations,
            estimated_maps=maps,
        )
        _validate_outputs(staging, config)
        primary = [row for row in recoveries if row["depth"] == symmetry["primary_depth"]]
        exact = sum(row["exact_recovery"] for row in primary)
        estimated = sum(row.get("estimated_recovery", False) for row in primary)
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
            f"seed={gate['permutation_seed']} logits={gate['maximum_logit_error']:.3e} "
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
    bundles = list((output / "probes").glob("permutation_*/seed_*/*.safetensors"))
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
