import os
from pathlib import Path
from typing import Any

from core.tracking import Tracker
from probe_transfer.alignment_evaluation import evaluate_checkpoint_alignment


def run_alignment_experiment(config: dict[str, Any], tracker: Tracker) -> None:
    baseline_dir = _required_directory("BASELINE_ARTIFACT_DIR")
    output_dir = _required_directory("EXPERIMENT_OUTPUT_DIR", create=True)
    if (output_dir / "results").exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir / 'results'}")

    recoveries, diagnostics, _ = evaluate_checkpoint_alignment(baseline_dir, output_dir, config)
    _assert_expected_outputs(output_dir, config)
    primary = _primary_recoveries(recoveries, config)
    if not primary:
        raise ValueError("No recovery rows matched the prespecified primary analysis.")

    for row in recoveries:
        prefix = (
            f"recovery/{row['source_model']}_to_{row['target_model']}/"
            f"seed_{row['data_seed']}/layer_{round(row['depth'] * 100)}/"
            f"{row['probe_family']}/{row['method']}"
        )
        values = {
            f"{prefix}/aligned_auroc_improvement": row["aligned_auroc_improvement"],
            f"{prefix}/residual_auroc_gap": row["residual_auroc_gap"],
        }
        if row["recovery_fraction"] is not None:
            values[f"{prefix}/recovery_fraction"] = row["recovery_fraction"]
        tracker.metrics(values)

    substantial = sum(bool(row["substantial_recovery"]) for row in primary)
    tracker.report(
        "Summary",
        f"Evaluated {len(recoveries)} recovery comparisons and {len(diagnostics)} held-out "
        f"alignment diagnostics. The primary restricted recovery rule passed "
        f"{substantial}/{len(primary)} prespecified comparisons.",
    )


def _primary_recoveries(
    recoveries: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    alignment = config["alignment"]
    evaluation = config["evaluation"]
    family = alignment.get("primary_probe_family")
    pair_group = evaluation.get("primary_pair_group")
    return [
        row
        for row in recoveries
        if row["depth"] == alignment["primary_depth"]
        and row["method"] == alignment["primary_method"]
        and (family is None or row["probe_family"] == family)
        and (pair_group is None or row["pair_group"] == pair_group)
    ]


def _assert_expected_outputs(output_dir: Path, config: dict[str, Any]) -> None:
    expected = config.get("expected_outputs")
    if expected is None:
        return
    paths = {
        "metrics_rows": output_dir / "results" / "metrics.jsonl",
        "prediction_rows": output_dir / "results" / "predictions.jsonl",
        "recovery_rows": output_dir / "results" / "recovery.jsonl",
        "alignment_diagnostic_rows": output_dir / "results" / "alignment_diagnostics.jsonl",
    }
    actual = {name: _line_count(path) for name, path in paths.items()}
    mismatches = {
        name: (expected[name], count)
        for name, count in actual.items()
        if expected.get(name) != count
    }
    if mismatches:
        raise ValueError(f"Unexpected output row counts: {mismatches}")


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


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
