import json
import math
import os
from pathlib import Path
from typing import Any

from core.constants import ACTIVATION_ROWS_ENV, BASELINE_ARTIFACT_ENV, EXPERIMENT_OUTPUT_ENV
from core.tracking import Tracker
from probe_transfer.artifacts import write_json, write_jsonl
from probe_transfer.data import load_prepared_rows
from probe_transfer.symmetry.evaluation import evaluate_permutations
from probe_transfer.symmetry.gate import run_function_gates
from probe_transfer.symmetry.transforms import seeded_permutation


def run_symmetry_experiment(config: dict[str, Any], tracker: Tracker) -> None:
    baseline = _required_directory(BASELINE_ARTIFACT_ENV)
    prepared = _required_directory(ACTIVATION_ROWS_ENV)
    output = _required_directory(EXPERIMENT_OUTPUT_ENV, create=True)
    for name in ("probes", "results"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output / name}")

    symmetry = config["symmetry"]
    permutations = {
        seed: seeded_permutation(symmetry["width"], seed) for seed in symmetry["permutation_seeds"]
    }
    write_json(
        output / "results" / "permutations.json",
        {str(seed): permutation.tolist() for seed, permutation in permutations.items()},
    )
    rows = load_prepared_rows(
        prepared / "test.jsonl", config["materials"]["expected_test_rows"], require_balanced=False
    )
    gates = run_function_gates(config, rows, permutations)
    write_jsonl(output / "results" / "function_gates.jsonl", gates)
    if any(not gate["passed"] for gate in gates):
        raise RuntimeError("At least one function-preservation gate failed.")

    recoveries, _ = evaluate_permutations(baseline, output, config, permutations)
    _validate_outputs(output, config)
    primary = [row for row in recoveries if row["depth"] == symmetry["primary_depth"]]
    tracker.report(
        "Summary",
        f"All {len(gates)} function-preservation gates passed; exact probe transport recovered "
        f"{sum(row['exact_recovery'] for row in primary)}/{len(primary)} primary comparisons.",
    )


def _validate_outputs(output: Path, config: dict[str, Any]) -> None:
    expected = config["expected_outputs"]
    paths = {
        "metrics_rows": output / "results" / "metrics.jsonl",
        "prediction_rows": output / "results" / "predictions.jsonl",
        "recovery_rows": output / "results" / "recovery.jsonl",
        "function_gate_rows": output / "results" / "function_gates.jsonl",
    }
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
