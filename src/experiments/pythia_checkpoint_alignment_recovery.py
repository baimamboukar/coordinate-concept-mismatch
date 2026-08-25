from typing import Any

from core.tracking import Tracker
from probe_transfer.alignment_runner import run_alignment_experiment


def run(config: dict[str, Any], tracker: Tracker) -> None:
    _validate_config(config)
    run_alignment_experiment(config, tracker)


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
