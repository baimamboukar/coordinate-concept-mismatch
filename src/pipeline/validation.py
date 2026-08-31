import re
from itertools import pairwise
from typing import Any

from core.config import ConfigError, validate_evaluation
from core.constants import HF_BUCKET
from pipeline.sweep_validation import validate_probe_sensitivity
from probe_transfer.alignment.cross_task import validate_cross_task_alignment
from probe_transfer.alignment.materials import direction_groups
from probe_transfer.alignment.selection import validate_alignment_selection
from probe_transfer.data import validate_prompt_configuration
from probe_transfer.extraction.sites import (
    ACTIVATION_SITES,
    ATTENTION_OUTPUT,
    MLP_INTERMEDIATE,
    RESIDUAL_STREAM,
    activation_width,
)
from probe_transfer.symmetry.protocol import estimated_alignment_enabled, selected_models

SCALE_VARIANT = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_stage(config: dict[str, Any]) -> None:
    _validate_artifacts(config)
    validate_prompt_configuration(config.get("dataset", {}))
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
    widths = {activation_width(config["activations"], model) for model in config["models"].values()}
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
    validate_cross_task_alignment(config)
    validate_alignment_selection(config)
    direction_groups(config)


def _validate_symmetry(config: dict[str, Any]) -> None:
    validate_evaluation(config.get("evaluation"))
    symmetry = config.get("symmetry", {})
    transformation = symmetry.get("transformation")
    site = config["activations"].get("site", RESIDUAL_STREAM)
    expected_site = {
        "residual_permutation": RESIDUAL_STREAM,
        "mlp_neuron_permutation": MLP_INTERMEDIATE,
        "mlp_positive_diagonal": MLP_INTERMEDIATE,
        "attention_head_permutation": ATTENTION_OUTPUT,
    }.get(transformation)
    if expected_site is None or site != expected_site:
        raise ConfigError("The symmetry transformation and activation site are incompatible.")
    models = selected_models(config)
    if not models or len(models) != len(set(models)) or set(models) - set(config["models"]):
        raise ConfigError("Symmetry models must be unique configured model keys.")
    if symmetry.get("primary_depth") not in symmetry.get("probed_depths", []):
        raise ConfigError("The primary symmetry depth must be evaluated.")
    if symmetry.get("gate_rows") != config["materials"].get("expected_test_rows"):
        raise ConfigError("Function gates must cover the complete protected test set.")
    if symmetry.get("gate_dtype") != "float64":
        raise ConfigError("Function-preservation gates must run in float64.")
    seeds = symmetry.get("transformation_seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ConfigError("At least one integer transformation seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ConfigError("Transformation seeds must be unique.")
    if symmetry.get("logit_atol", 0) <= 0 or symmetry.get("logit_rtol", 0) <= 0:
        raise ConfigError("Function-preservation tolerances must be positive.")
    widths = {activation_width(config["activations"], config["models"][name]) for name in models}
    if widths != {symmetry.get("width")}:
        raise ConfigError("Every transformed model must match the configured activation width.")
    if transformation == "attention_head_permutation":
        _validate_attention_layout(config, models)
    elif transformation == "mlp_positive_diagonal":
        _validate_positive_diagonal(config)
    smoke_rows = symmetry.get("smoke_gate_rows", 0)
    if smoke_rows < 0 or smoke_rows >= symmetry["gate_rows"]:
        raise ConfigError("Symmetry smoke rows must be non-negative and smaller than gate rows.")
    if estimated_alignment_enabled(config):
        settings = symmetry["estimated_alignment"]
        materials = config.get("materials", {})
        expected_method = (
            {"positive_diagonal"}
            if transformation == "mlp_positive_diagonal"
            else {"permutation", "exact_permutation"}
        )
        if settings.get("method") not in expected_method:
            raise ConfigError("Known-symmetry estimation requires the matching strict method.")
        if settings.get("fit_split") != "train" or settings.get("diagnostic_split") != "validation":
            raise ConfigError(
                "Estimated symmetry alignment requires train fit and validation diagnosis."
            )
        fit_rows = settings.get("fit_rows", 0)
        if fit_rows < 2 or fit_rows > materials.get("expected_train_rows", 0):
            raise ConfigError("Estimated alignment fit rows exceed the training material contract.")
        if materials.get("expected_validation_rows") != config["sampling"]["validation_size"]:
            raise ConfigError("Estimated alignment validation rows must match the protected split.")
        if settings.get("device") not in {"cpu", "cuda", "auto"}:
            raise ConfigError("Estimated alignment device must be cpu, cuda, or auto.")


def _validate_positive_diagonal(config: dict[str, Any]) -> None:
    symmetry = config["symmetry"]
    scale_range = symmetry.get("scale_range")
    scale_ranges = symmetry.get("scale_ranges")
    if (scale_range is None) == (scale_ranges is None):
        raise ConfigError("Configure exactly one positive-diagonal scale range or range sweep.")
    if scale_ranges is not None:
        if (
            not isinstance(scale_ranges, dict)
            or len(scale_ranges) < 2
            or any(
                not isinstance(name, str) or not SCALE_VARIANT.fullmatch(name)
                for name in scale_ranges
            )
        ):
            raise ConfigError("Scale sweeps require at least two semantic lowercase variants.")
        ranges = list(scale_ranges.values())
    else:
        ranges = [scale_range]
    for values in ranges:
        _validate_reciprocal_range(values)
    if scale_ranges is not None:
        maxima = [values[1] for values in ranges]
        if any(after <= before for before, after in pairwise(maxima)):
            raise ConfigError("Scale sweep ranges must increase in configured order.")
        _validate_dose_response(config, list(scale_ranges))
    elif symmetry.get("dose_response") is not None:
        raise ConfigError("Dose-response settings require a positive-diagonal scale sweep.")

    settings = symmetry.get("estimated_alignment", {})
    if settings.get("fit_relative_tolerance", 0) <= 0 or settings.get("scale_match_rtol", 0) <= 0:
        raise ConfigError("Positive-diagonal estimation tolerances must be positive.")


def _validate_reciprocal_range(scale_range: Any) -> None:
    if (
        not isinstance(scale_range, list)
        or len(scale_range) != 2
        or any(type(value) not in {int, float} for value in scale_range)
    ):
        raise ConfigError("Positive-diagonal symmetry requires a two-value scale range.")
    minimum, maximum = scale_range
    if minimum <= 0 or minimum >= 1 or maximum <= 1 or abs(minimum * maximum - 1) > 1e-12:
        raise ConfigError("Positive-diagonal scales must use a reciprocal range around one.")


def _validate_dose_response(config: dict[str, Any], variants: list[str]) -> None:
    settings = config["symmetry"].get("dose_response")
    if not isinstance(settings, dict) or settings.get("ordered_variants") != variants:
        raise ConfigError("Dose-response variants must exactly match the configured range order.")
    minimum_rho = settings.get("minimum_trajectory_spearman")
    if not isinstance(minimum_rho, (int, float)) or isinstance(minimum_rho, bool):
        raise ConfigError(
            "The minimum trajectory Spearman correlation must be between zero and one."
        )
    if not 0 <= minimum_rho <= 1:
        raise ConfigError(
            "The minimum trajectory Spearman correlation must be between zero and one."
        )
    primary = len(config["probes"]["primary_families"])
    comparisons = (
        len(config["data_seeds"])
        * len(selected_models(config))
        * len(config["symmetry"]["transformation_seeds"])
        * primary
    )
    for key in ("minimum_monotonic_trajectories", "minimum_failure_comparisons"):
        value = settings.get(key)
        if type(value) is not int or not 1 <= value <= comparisons:
            raise ConfigError(f"{key} must be an integer between one and {comparisons}.")
    validate_probe_sensitivity(config, variants)


def _validate_attention_layout(config: dict[str, Any], models: list[str]) -> None:
    symmetry = config["symmetry"]
    layout = symmetry.get("attention_layout")
    keys = ("query_heads", "key_value_heads", "head_dim")
    if not isinstance(layout, dict) or any(type(layout.get(key)) is not int for key in keys):
        raise ConfigError("Attention symmetry requires an integer attention_layout.")
    query_heads, key_value_heads, head_dim = (layout[key] for key in keys)
    if min(query_heads, key_value_heads, head_dim) < 1 or query_heads % key_value_heads:
        raise ConfigError("Attention symmetry requires a valid grouped-query head layout.")
    if query_heads * head_dim != symmetry["width"]:
        raise ConfigError("Attention head layout must match the probed activation width.")
    expected = {
        "attention_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_dim": head_dim,
    }
    for name in models:
        model = config["models"][name]
        if any(model.get(key) != value for key, value in expected.items()):
            raise ConfigError(f"Attention layout does not match configured model {name}.")


def _validate_activation_protocol(config: dict[str, Any]) -> None:
    activations = config["activations"]
    if activations.get("site", RESIDUAL_STREAM) not in ACTIVATION_SITES:
        raise ConfigError("The activation site is unsupported.")
    depths = activations.get("normalized_depths", [])
    if not depths or len(depths) != len(set(depths)):
        raise ConfigError("Normalized activation depths must be non-empty and unique.")
    if activations.get("primary_depth") not in depths:
        raise ConfigError("The primary depth must be extracted.")
    if activations.get("prompt_format") != "raw":
        raise ConfigError("The current pipeline requires raw prompts.")
    if activations.get("token_position") != "last_non_padding":
        raise ConfigError("The current pipeline requires last-non-padding activations.")
    if not isinstance(activations.get("add_special_tokens", True), bool):
        raise ConfigError("add_special_tokens must be a boolean.")
