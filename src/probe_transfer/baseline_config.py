from typing import Any

PHASE_MODELS = {
    "core": ("llama", "qwen", "nemotron"),
    "cross_family_extension": ("llama", "qwen", "nemotron", "mistral", "granite"),
}
PHASE_EXTRACTIONS = {
    "core": ("llama", "qwen", "nemotron"),
    "cross_family_extension": ("mistral", "granite"),
}


def validate_modern_baseline(config: dict[str, Any]) -> None:
    phase = config.get("phase", "core")
    if phase not in PHASE_MODELS:
        raise ValueError(f"Unsupported modern baseline phase: {phase}")
    models = list(config["models"])
    if tuple(models) != PHASE_MODELS[phase]:
        raise ValueError(f"The {phase} model order changed.")
    if {model["hidden_size"] for model in config["models"].values()} != {4096}:
        raise ValueError("Direct transfer requires 4,096-dimensional residual streams.")

    sampling = config["sampling"]
    if sampling.get("test_size") != 1699:
        raise ValueError("The modern baseline must reuse the protected 1,699-row test set.")
    if sampling["train_size"] % 2 or sampling["validation_size"] % 2:
        raise ValueError("Train and validation sizes must be even.")
    if (
        sampling.get("balance_labels") is not True
        or sampling.get("stratify_by") != ["adversarial"]
        or sampling.get("protect_test") is not True
    ):
        raise ValueError("The modern baseline requires balanced, stratified, protected splits.")

    activations = config["activations"]
    depths = activations["normalized_depths"]
    if depths != [0.25, 0.5, 0.75, 1.0] or activations["primary_depth"] != 0.75:
        raise ValueError("The modern baseline requires the four prespecified normalized depths.")
    if activations["prompt_format"] != "raw" or activations["token_position"] != "last_non_padding":
        raise ValueError("Only raw prompts and last-non-padding activations are supported.")

    extraction = config["extraction"]
    expected_extractions = list(PHASE_EXTRACTIONS[phase])
    if extraction.get("mode") != "full" or extraction.get("models") != expected_extractions:
        raise ValueError(f"The {phase} extraction model contract changed.")
    jobs = extraction.get("jobs", [])
    if {job.get("model") for job in jobs} != set(expected_extractions) or len(jobs) != len(
        expected_extractions
    ):
        raise ValueError("The modern baseline requires exactly one extraction job per model.")
    if any(job.get("accelerator") != "H100" or job.get("gpu_count") != 1 for job in jobs):
        raise ValueError("Each modern extraction job requires one H100.")

    if config.get("training") is not True or config.get("tracking") != {
        "wandb": True,
        "mode": "offline",
    }:
        raise ValueError("The modern baseline requires offline W&B tracking for probe training.")
    artifacts = config["artifacts"]
    if (
        artifacts.get("backend") != "huggingface_bucket"
        or artifacts.get("defer_upload") is not True
    ):
        raise ValueError("Workers must stage Hugging Face artifacts without uploading them.")

    groups = config["evaluation"]["pair_groups"]
    assigned = []
    for group, pairs in groups.items():
        for source, target in pairs:
            if source not in models or target not in models or source == target:
                raise ValueError(f"Invalid {group} transfer pair: {source} -> {target}")
            assigned.append((source, target))
    expected = {(source, target) for source in models for target in models if source != target}
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError("Every directed model pair must belong to exactly one comparison group.")
    claim = "broad_three_family" if phase == "core" else "broad_cross_family"
    if config.get("claims", {}).get(claim) != "pending":
        raise ValueError(f"The {claim} claim must remain pending until evaluation.")

    execution = config.get("execution")
    if execution != {
        "accelerator": "H100",
        "gpu_count": 1,
        "minimum_cuda_driver_support": 13.0,
        "minimum_gpu_memory_gb": 75,
        "minimum_disk_free_gb": 50,
    }:
        raise ValueError("The modern extraction runtime contract changed.")

    family_depths = len(config["data_seeds"]) * (
        len(activations["normalized_depths"]) - 1 + len(config["probes"]["primary_families"])
    )
    model_count = len(models)
    expected_outputs = {
        "metrics_rows": family_depths * model_count**2,
        "prediction_rows": family_depths * model_count**2 * sampling["test_size"],
        "transfer_gap_rows": family_depths * model_count * (model_count - 1),
        "probe_bundles": len(config["data_seeds"]) * model_count,
    }
    if config.get("expected_outputs") != expected_outputs:
        raise ValueError("The expected modern baseline output counts changed.")
