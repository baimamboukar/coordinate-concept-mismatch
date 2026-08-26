from typing import Any

from core.config import ConfigError, validate_evaluation
from core.constants import HF_BUCKET
from probe_transfer.alignment.materials import direction_groups


def validate_stage(config: dict[str, Any]) -> None:
    _validate_artifacts(config)
    if config.get("training") and not config.get("tracking", {}).get("wandb"):
        raise ConfigError("Training stages must enable W&B tracking.")
    stage = config["stage"]
    if stage in {"preflight", "extract"}:
        _validate_extraction(config)
    elif stage == "transfer":
        _validate_transfer(config)
    elif stage == "align":
        _validate_alignment(config)
    elif stage == "symmetry":
        _validate_symmetry(config)
    elif stage != "prepare":
        raise ConfigError(f"Unsupported pipeline stage: {stage}")


def _validate_artifacts(config: dict[str, Any]) -> None:
    artifacts = config.get("artifacts", {})
    if artifacts.get("backend") != "huggingface_bucket":
        raise ConfigError("Artifacts must use the Hugging Face bucket backend.")
    if artifacts.get("bucket") != HF_BUCKET:
        raise ConfigError(f"Artifacts must use the project bucket: {HF_BUCKET}")
    if artifacts.get("publish_from_worker") is not True:
        raise ConfigError("Completed artifacts must publish directly from the worker.")
    if artifacts.get("verify_anonymously") is not True:
        raise ConfigError("Worker publication must be anonymously verified.")


def _validate_extraction(config: dict[str, Any]) -> None:
    extraction = config.get("extraction", {})
    models = extraction.get("models")
    if extraction.get("mode") != "full" or not isinstance(models, list) or not models:
        raise ConfigError("Extraction requires a non-empty extraction.models list in full mode.")
    if len(models) != len(set(models)) or set(models) - set(config["models"]):
        raise ConfigError("Extraction models must be unique configured model keys.")
    _validate_activation_protocol(config)


def _validate_transfer(config: dict[str, Any]) -> None:
    _validate_activation_protocol(config)
    validate_evaluation(config.get("evaluation"))
    widths = {model["hidden_size"] for model in config["models"].values()}
    if len(widths) != 1:
        raise ConfigError("Unaligned frozen transfer requires a shared activation width.")
    models = set(config["models"])
    pairs = []
    for group, directions in config["evaluation"].get("pair_groups", {}).items():
        for source, target in directions:
            if source not in models or target not in models or source == target:
                raise ConfigError(f"Invalid {group} transfer direction: {source} -> {target}")
            pairs.append((source, target))
    expected = {(source, target) for source in models for target in models if source != target}
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ConfigError("Every directed transfer pair must belong to exactly one group.")


def _validate_alignment(config: dict[str, Any]) -> None:
    validate_evaluation(config.get("evaluation"))
    alignment = config.get("alignment", {})
    methods = alignment.get("methods", [])
    supported = {
        "permutation",
        "permutation_diagonal",
        "orthogonal_procrustes",
        "affine_ridge",
        "quotient_ridge",
    }
    if not methods or len(methods) != len(set(methods)) or set(methods) - supported:
        raise ConfigError("Alignment methods must be unique supported methods.")
    if alignment.get("primary_method") not in methods:
        raise ConfigError("The primary alignment method must be evaluated.")
    if alignment.get("primary_depth") not in alignment.get("depths", []):
        raise ConfigError("The primary alignment depth must be evaluated.")
    if alignment.get("negative_control") in methods:
        raise ConfigError("The alignment negative control must be separate from fitted methods.")
    if alignment.get("negative_control") != "shuffled_affine_ridge":
        raise ConfigError("Alignment requires the shuffled-pair affine negative control.")
    if alignment.get("primary_probe_family") not in config["probes"]["primary_families"]:
        raise ConfigError("The primary alignment probe family must be trained at primary depth.")
    if len({model["hidden_size"] for model in config["models"].values()}) != 1:
        raise ConfigError("The configured alignment maps require a shared activation width.")
    materials = config.get("materials", {})
    sampling = config["sampling"]
    for split in ("train", "validation", "test"):
        if materials.get(f"expected_{split}_rows") != sampling[f"{split}_size"]:
            raise ConfigError(f"Alignment {split} rows must match the prepared split contract.")
    direction_groups(config)


def _validate_symmetry(config: dict[str, Any]) -> None:
    validate_evaluation(config.get("evaluation"))
    symmetry = config.get("symmetry", {})
    if symmetry.get("primary_depth") not in symmetry.get("probed_depths", []):
        raise ConfigError("The primary symmetry depth must be evaluated.")
    if symmetry.get("gate_rows") != config["materials"].get("expected_test_rows"):
        raise ConfigError("Function gates must cover the complete protected test set.")
    if symmetry.get("gate_dtype") != "float64":
        raise ConfigError("Function-preservation gates must run in float64.")
    if not symmetry.get("permutation_seeds"):
        raise ConfigError("At least one permutation seed is required.")
    if len(symmetry["permutation_seeds"]) != len(set(symmetry["permutation_seeds"])):
        raise ConfigError("Permutation seeds must be unique.")
    if symmetry.get("logit_atol", 0) <= 0 or symmetry.get("logit_rtol", 0) <= 0:
        raise ConfigError("Function-preservation tolerances must be positive.")
    widths = {model["hidden_size"] for model in config["models"].values()}
    if widths != {symmetry.get("width")}:
        raise ConfigError("Every transformed model must match the configured residual width.")


def _validate_activation_protocol(config: dict[str, Any]) -> None:
    activations = config["activations"]
    depths = activations.get("normalized_depths", [])
    if not depths or len(depths) != len(set(depths)):
        raise ConfigError("Normalized activation depths must be non-empty and unique.")
    if activations.get("primary_depth") not in depths:
        raise ConfigError("The primary depth must be extracted.")
    if activations.get("prompt_format") != "raw":
        raise ConfigError("The current pipeline requires raw prompts.")
    if activations.get("token_position") != "last_non_padding":
        raise ConfigError("The current pipeline requires last-non-padding activations.")
