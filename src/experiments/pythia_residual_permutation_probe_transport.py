import os
from pathlib import Path
from typing import Any

from core.tracking import Tracker
from probe_transfer.artifacts import write_json, write_jsonl
from probe_transfer.data import load_prepared_rows
from probe_transfer.function_gate import run_function_gates
from probe_transfer.symmetry import seeded_permutation
from probe_transfer.symmetry_evaluation import evaluate_permutations


def run(config: dict[str, Any], tracker: Tracker) -> None:
    _validate_config(config)
    baseline_dir = _required_directory("BASELINE_ARTIFACT_DIR")
    prepared_dir = _required_directory("PREPARED_DATA_DIR")
    output_dir = _required_directory("EXPERIMENT_OUTPUT_DIR", create=True)
    _assert_new_output(output_dir)

    symmetry = config["symmetry"]
    permutations = {
        seed: seeded_permutation(symmetry["width"], seed) for seed in symmetry["permutation_seeds"]
    }
    write_json(
        output_dir / "results" / "permutations.json",
        {str(seed): permutation.tolist() for seed, permutation in permutations.items()},
    )

    rows = load_prepared_rows(
        prepared_dir / "test.jsonl",
        config["materials"]["expected_test_rows"],
        require_balanced=False,
    )[: symmetry["gate_rows"]]
    gates = run_function_gates(config, rows, permutations)
    write_jsonl(output_dir / "results" / "function_gates.jsonl", gates)
    _log_gates(tracker, gates)
    failed_gates = [gate for gate in gates if not gate["passed"]]
    if failed_gates:
        raise RuntimeError(f"{len(failed_gates)} function-preservation gates failed.")

    recoveries, _ = evaluate_permutations(baseline_dir, output_dir, config, permutations)
    _log_recoveries(tracker, recoveries)
    primary = [row for row in recoveries if row["depth"] == symmetry["primary_depth"]]
    coordinate_failures = sum(row["coordinate_failure"] for row in primary)
    exact_recoveries = sum(row["exact_recovery"] for row in primary)
    tracker.report(
        "Summary",
        f"All {len(gates)} function-preservation gates passed. Coordinate-induced failure "
        f"held for {coordinate_failures}/{len(primary)} primary comparisons and exact "
        f"recovery held for {exact_recoveries}/{len(primary)}.",
    )


def _log_gates(tracker: Tracker, gates: list[dict[str, Any]]) -> None:
    for gate in gates:
        suffix = gate["permutation_seed"] if gate["permutation_seed"] is not None else "identity"
        prefix = f"gates/{gate['model']}/{suffix}"
        tracker.metrics(
            {
                f"{prefix}/maximum_logit_error": gate["maximum_logit_error"],
                f"{prefix}/next_token_agreement": gate["next_token_agreement"],
                f"{prefix}/maximum_activation_relative_error": gate[
                    "maximum_activation_relative_error"
                ],
            }
        )


def _log_recoveries(tracker: Tracker, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        prefix = (
            f"recovery/{row['model']}/data_{row['data_seed']}/perm_{row['permutation_seed']}/"
            f"layer_{round(row['depth'] * 100)}/{row['probe_family']}"
        )
        tracker.metrics(
            {
                f"{prefix}/raw_auroc_gap": row["raw_auroc_gap"],
                f"{prefix}/recovery_fraction": row["recovery_fraction"] or 0.0,
                f"{prefix}/maximum_score_error": row["maximum_score_error"],
            }
        )


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


def _assert_new_output(output_dir: Path) -> None:
    for name in ("probes", "results"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir / name}")


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "controlled_residual_permutation":
        raise ValueError("The experiment requires its controlled residual-permutation stage.")
    if config.get("training") is not False:
        raise ValueError("The controlled experiment must reuse probes without retraining.")
    if len(config["models"]) != 2 or len(config["data_seeds"]) != 2:
        raise ValueError("The experiment requires two checkpoints and two data seeds.")
    symmetry = config["symmetry"]
    if symmetry["gate_rows"] != config["materials"]["expected_test_rows"]:
        raise ValueError("Function gates must cover the complete held-out test set.")
    if symmetry["permutation_seeds"] != [42, 137]:
        raise ValueError("The prespecified permutation seeds are 42 and 137.")
    widths = {model["hidden_size"] for model in config["models"].values()}
    if widths != {symmetry["width"]}:
        raise ValueError("Every model must match the configured residual width.")
    if symmetry["primary_depth"] not in symmetry["probed_depths"]:
        raise ValueError("The primary depth must be included in the function gates.")
    if config["artifacts"].get("defer_upload") is not True:
        raise ValueError("The GPU worker must stage artifacts before verified upload.")
