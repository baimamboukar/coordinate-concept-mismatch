from typing import Any

from core.tracking import Tracker
from probe_transfer.alignment_materials import direction_groups
from probe_transfer.alignment_runner import run_alignment_experiment

PHASE_CONTRACTS = {
    "core": {
        "models": ["llama", "qwen", "nemotron"],
        "groups": {
            "primary": [["llama", "qwen"], ["qwen", "llama"]],
            "lineage_control": [["llama", "nemotron"], ["nemotron", "llama"]],
            "exploratory": [["qwen", "nemotron"], ["nemotron", "qwen"]],
        },
        "materials": {
            "baseline_artifact_prefix": "experiments/frozen_probe_transfer_baseline/modern_phase",
            "expected_train_rows": 12000,
            "expected_validation_rows": 2000,
            "expected_test_rows": 1699,
        },
        "outputs": {
            "metrics_rows": 264,
            "prediction_rows": 448536,
            "recovery_rows": 192,
            "alignment_diagnostic_rows": 72,
        },
        "prefix": "experiments/modern_activation_alignment_recovery",
    },
    "cross_family_extension": {
        "models": ["llama", "qwen", "mistral", "granite"],
        "groups": {
            "primary": [
                ["llama", "mistral"],
                ["mistral", "llama"],
                ["qwen", "mistral"],
                ["mistral", "qwen"],
                ["llama", "granite"],
                ["granite", "llama"],
                ["qwen", "granite"],
                ["granite", "qwen"],
                ["mistral", "granite"],
                ["granite", "mistral"],
            ]
        },
        "materials": {
            "baseline_artifact_prefixes": [
                "experiments/frozen_probe_transfer_baseline/modern_phase",
                "experiments/frozen_probe_transfer_baseline/cross_family_extension",
            ],
            "expected_train_rows": 12000,
            "expected_validation_rows": 2000,
            "expected_test_rows": 1699,
        },
        "outputs": {
            "metrics_rows": 440,
            "prediction_rows": 747560,
            "recovery_rows": 320,
            "alignment_diagnostic_rows": 120,
        },
        "prefix": "experiments/modern_activation_alignment_recovery/cross_family_extension",
    },
}


def run(config: dict[str, Any], tracker: Tracker) -> None:
    _validate_config(config)
    run_alignment_experiment(config, tracker)


def _validate_config(config: dict[str, Any]) -> None:
    phase = config.get("phase", "core")
    if phase not in PHASE_CONTRACTS:
        raise ValueError(f"Unsupported modern alignment phase: {phase}")
    contract = PHASE_CONTRACTS[phase]
    models = list(config["models"])
    if config.get("stage") != "modern_natural_alignment" or config.get("training") is not False:
        raise ValueError("This experiment must reuse frozen modern-model probes.")
    if models != contract["models"] or config["data_seeds"] != [42, 137]:
        raise ValueError(f"The {phase} alignment model or seed contract changed.")
    if {model["hidden_size"] for model in config["models"].values()} != {4096}:
        raise ValueError("The modern alignment study requires 4,096-dimensional activations.")
    if any(not model.get("revision") for model in config["models"].values()):
        raise ValueError("Every checkpoint revision must be pinned.")
    if config["materials"] != contract["materials"]:
        raise ValueError("The verified modern baseline material contract changed.")

    alignment = config["alignment"]
    required_methods = {
        "permutation",
        "permutation_diagonal",
        "orthogonal_procrustes",
        "affine_ridge",
        "quotient_ridge",
    }
    if (
        alignment["depths"] != [0.75]
        or alignment["primary_depth"] != 0.75
        or alignment["primary_probe_family"] != "linear"
        or alignment["primary_method"] != "permutation_diagonal"
        or set(alignment["methods"]) != required_methods
        or alignment["negative_control"] != "shuffled_affine_ridge"
        or alignment["fit_split"] != "train"
        or alignment["diagnostic_split"] != "validation"
        or alignment["device"] != "cuda"
    ):
        raise ValueError("The prespecified modern alignment method or primary analysis changed.")

    evaluation = config["evaluation"]
    if evaluation.get("pair_groups") != contract["groups"]:
        raise ValueError("The prespecified modern pair groups changed.")
    if evaluation.get("primary_pair_group") != "primary":
        raise ValueError("Llama-Qwen must remain the primary model pair.")
    if evaluation.get("primary_metrics") != [
        "aligned_auroc_improvement",
        "recovery_fraction",
        "residual_auroc_gap",
    ]:
        raise ValueError("The primary metric contract changed.")
    required_secondary = {
        "auroc",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "expected_calibration_error",
        "tn",
        "fp",
        "fn",
        "tp",
        "tpr_at_fpr",
        "achieved_fpr_at_source_threshold",
        "alignment_relative_rmse",
        "alignment_mean_cosine",
        "negative_control_results",
    }
    if not required_secondary.issubset(evaluation["secondary_metrics"]):
        raise ValueError("The secondary metric contract is incomplete.")
    expected_directions = {tuple(pair) for pairs in contract["groups"].values() for pair in pairs}
    actual_directions = {(source, target) for source, target, _ in direction_groups(config)}
    if actual_directions != expected_directions:
        raise ValueError("Every directed modern-model pair must be evaluated exactly once.")
    if evaluation["retain_row_level"] != [
        "row_id",
        "label",
        "score",
        "probability",
        "prediction",
    ]:
        raise ValueError("The row-level evidence contract changed.")
    if (
        evaluation["operating_fprs"] != [0.01, 0.05]
        or evaluation["retain_thresholds"] is not True
        or evaluation["bootstrap_samples"] != 2000
        or evaluation["confidence_level"] != 0.95
        or evaluation["require_all_primary_comparisons"] is not True
        or evaluation["lineage_non_degradation_margin"] != 0.02
    ):
        raise ValueError("The threshold, uncertainty, or lineage-control contract changed.")

    if config.get("expected_outputs") != contract["outputs"]:
        raise ValueError("The expected primary-depth output counts changed.")
    if config.get("tracking") != {"wandb": True, "mode": "offline"}:
        raise ValueError("Alignment fitting must use offline W&B tracking.")
    if config.get("execution") != {
        "accelerator": "H100",
        "gpu_count": 1,
        "minimum_cuda_driver_support": 13.0,
    }:
        raise ValueError("The modern alignment run requires one CUDA-13-compatible H100.")
    artifacts = config["artifacts"]
    if (
        artifacts.get("backend") != "huggingface_bucket"
        or artifacts.get("bucket") != "baimamboukar/coordinate-concept-mismatch"
        or artifacts.get("prefix") != contract["prefix"]
        or artifacts.get("defer_upload") is not True
    ):
        raise ValueError("Verified alignment artifacts must be staged for the project HF bucket.")
