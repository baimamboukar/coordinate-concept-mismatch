from typing import Any


def validate_modern_baseline(config: dict[str, Any]) -> None:
    models = list(config["models"])
    if models != ["llama", "qwen", "nemotron"]:
        raise ValueError("The current modern phase requires Llama, Qwen, and Nemotron in order.")
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
    if extraction.get("mode") != "full" or extraction.get("models") != models:
        raise ValueError("The modern baseline must fully extract every configured model in order.")
    jobs = extraction.get("jobs", [])
    if {job.get("model") for job in jobs} != set(models) or len(jobs) != len(models):
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
    for group in ("primary", "lineage_control", "exploratory"):
        for source, target in groups.get(group, []):
            if source not in models or target not in models or source == target:
                raise ValueError(f"Invalid {group} transfer pair: {source} -> {target}")
            assigned.append((source, target))
    expected = {(source, target) for source in models for target in models if source != target}
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError("Every directed model pair must belong to exactly one comparison group.")
    if config.get("claims", {}).get("broad_three_family") != "pending":
        raise ValueError("The three-family claim must remain pending until Mistral is evaluated.")
