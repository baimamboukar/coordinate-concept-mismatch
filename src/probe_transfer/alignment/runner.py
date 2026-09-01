import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from core.constants import BASELINE_ARTIFACT_ENV, EXPERIMENT_OUTPUT_ENV
from core.tracking import Tracker
from probe_transfer.alignment.evaluation import evaluate_checkpoint_alignment
from probe_transfer.atomic import publish_directories


def run_alignment_experiment(
    config: dict[str, Any],
    tracker: Tracker,
    *,
    fit_root: Path | None = None,
    reference_path: Path | None = None,
) -> None:
    baseline_dir = _required_directory(BASELINE_ARTIFACT_ENV)
    output_dir = _required_directory(EXPERIMENT_OUTPUT_ENV, create=True)
    if (output_dir / "results").exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir / 'results'}")

    with TemporaryDirectory(prefix=".alignment-", dir=output_dir) as temporary:
        staging = Path(temporary)
        recoveries, diagnostics, _ = evaluate_checkpoint_alignment(
            baseline_dir,
            staging,
            config,
            fit_root=fit_root,
            reference_path=reference_path,
        )
        _assert_expected_outputs(staging, config)
        primary = _primary_recoveries(recoveries, config)
        if not primary:
            raise ValueError("No recovery rows matched the prespecified primary analysis.")

        selection_path = staging / "results" / "alignment_selection.jsonl"
        if selection_path.is_file():
            for line in selection_path.read_text().splitlines():
                row = json.loads(line)
                if row["selected"]:
                    prefix = (
                        f"fitting/{row['source_model']}_to_{row['target_model']}/"
                        f"seed_{row['data_seed']}/{row['method']}/{row['fit_task']}"
                    )
                    keys = (
                        "relative_alpha",
                        "source_variance_power",
                        "validation_relative_mse",
                        "validation_probe_score_relative_mse",
                        "sample_weight",
                        "weighted_train_loss_fraction",
                    )
                    tracker.metrics(
                        {f"{prefix}/{key}": row[key] for key in keys if row.get(key) is not None}
                    )

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
        publish_directories(staging, output_dir, ("results",))


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
    if "alignment_selection_rows" in expected:
        paths["alignment_selection_rows"] = output_dir / "results" / "alignment_selection.jsonl"
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
