import os
from pathlib import Path
from typing import Any

from core.tracking import Tracker
from probe_transfer.alignment_evaluation import evaluate_checkpoint_alignment


def run(config: dict[str, Any], tracker: Tracker) -> None:
    _validate_config(config)
    baseline_dir = _required_directory("BASELINE_ARTIFACT_DIR")
    output_dir = _required_directory("EXPERIMENT_OUTPUT_DIR", create=True)
    if (output_dir / "results").exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir / 'results'}")

    recoveries, diagnostics, _ = evaluate_checkpoint_alignment(baseline_dir, output_dir, config)
    primary = [
        row
        for row in recoveries
        if row["depth"] == config["alignment"]["primary_depth"]
        and row["method"] == config["alignment"]["primary_method"]
    ]
    for row in recoveries:
        prefix = (
            f"recovery/{row['source_model']}_to_{row['target_model']}/"
            f"seed_{row['data_seed']}/layer_{round(row['depth'] * 100)}/"
            f"{row['probe_family']}/{row['method']}"
        )
        tracker.metrics(
            {
                f"{prefix}/aligned_auroc_improvement": row["aligned_auroc_improvement"],
                f"{prefix}/recovery_fraction": row["recovery_fraction"] or 0.0,
                f"{prefix}/residual_auroc_gap": row["residual_auroc_gap"],
            }
        )
    substantial = sum(row["substantial_recovery"] for row in primary)
    tracker.report(
        "Summary",
        f"Evaluated {len(recoveries)} alignment comparisons and {len(diagnostics)} held-out "
        f"alignment diagnostics. The primary restricted map substantially recovered "
        f"{substantial}/{len(primary)} comparisons.",
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


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "natural_checkpoint_alignment" or config.get("training") is not False:
        raise ValueError("This experiment must reuse frozen checkpoint probes.")
    if len(config["models"]) != 2 or config["data_seeds"] != [42, 137]:
        raise ValueError(
            "The experiment requires two Pythia checkpoints and data seeds 42 and 137."
        )
    alignment = config["alignment"]
    required = {
        "permutation",
        "permutation_diagonal",
        "orthogonal_procrustes",
        "affine_ridge",
        "quotient_ridge",
    }
    if set(alignment["methods"]) != required:
        raise ValueError("The prespecified alignment method set changed.")
    if alignment["primary_method"] != "permutation_diagonal":
        raise ValueError("Permutation-diagonal matching is the primary restricted alignment.")
    if alignment["negative_control"] != "shuffled_affine_ridge":
        raise ValueError("Shuffled-pair affine Ridge is the required negative control.")
    if alignment["primary_depth"] not in alignment["depths"]:
        raise ValueError("The primary depth must be evaluated.")
    if config["artifacts"].get("defer_upload") is not True:
        raise ValueError("The remote worker must stage artifacts before verified upload.")
