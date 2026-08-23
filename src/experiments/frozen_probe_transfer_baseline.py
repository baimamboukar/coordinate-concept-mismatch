import gc
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from core.tracking import Tracker
from probe_transfer.activations import (
    assert_repeatable,
    extract_activation_tensors,
    save_activation_file,
    upload_bucket_file,
)
from probe_transfer.data import balanced_subset, load_huggingface_dataset, prepare_splits
from probe_transfer.models import load_activation_model


def run(config: dict[str, Any], tracker: Tracker) -> None:
    _validate_config(config)
    stage = config.get("stage")
    if stage == "prepare_data":
        _prepare_data(config, tracker)
    elif stage == "extract_activations":
        _extract_activation_smoke(config, tracker)
    else:
        raise ValueError(f"Unsupported baseline stage: {stage}")


def _prepare_data(config: dict[str, Any], tracker: Tracker) -> None:
    test, seed_splits, audit = _load_splits(config)
    dataset = config["dataset"]
    seed_details = "\n\n".join(
        _seed_report(seed, splits) for seed, splits in sorted(seed_splits.items())
    )
    test_counts = _counts(test)
    tracker.report(
        "Data",
        f"Source: `allenai/wildguardmix` at `{dataset['revision']}`.\n\n"
        f"Clean test: {test_counts['rows']:,} rows "
        f"({test_counts['labels'][0]:,} unharmful, {test_counts['labels'][1]:,} harmful).\n\n"
        f"- Removed: {audit['train'].get('duplicate_prompt', 0):,} duplicate train rows, "
        f"{audit['train'].get('conflicting_label_prompts', 0)} conflicting-label prompts, "
        f"and {audit['test'].get('invalid_label', 0)} unlabeled test rows.\n"
        "- No model inference or probe training was run.\n\n"
        f"{seed_details}",
    )


def _extract_activation_smoke(config: dict[str, Any], tracker: Tracker) -> None:
    _, seed_splits, _ = _load_splits(config)
    extraction = config["extraction"]
    if extraction["mode"] != "smoke":
        raise ValueError("Only smoke activation extraction is implemented.")

    seed = extraction["seed"]
    rows = balanced_subset(seed_splits[seed]["train"], extraction["sample_size"], seed)
    activation_config = config["activations"]
    artifact_config = config["artifacts"]
    extracted = []

    with TemporaryDirectory(prefix="coordinate-concept-activations-") as temporary:
        temporary_dir = Path(temporary)
        for model_name in extraction["models"]:
            model_config = config["models"][model_name]
            tokenizer, model = load_activation_model(
                model_config["id"],
                model_config["revision"],
                dtype=activation_config["dtype"],
            )
            tensors, stats = _extract(config, rows, tokenizer, model, model_config)
            if stats.truncation_rate > activation_config["max_truncation_rate"]:
                raise ValueError(
                    f"{model_name} truncation rate {stats.truncation_rate:.2%} exceeds "
                    f"{activation_config['max_truncation_rate']:.2%}."
                )

            repeated, _ = _extract(
                config,
                rows[: extraction["repeatability_rows"]],
                tokenizer,
                model,
                model_config,
            )
            assert_repeatable(
                tensors,
                repeated,
                rows=extraction["repeatability_rows"],
                atol=extraction["repeatability_atol"],
            )
            path = temporary_dir / f"{model_name}.safetensors"
            sha256 = save_activation_file(
                path,
                tensors,
                _activation_metadata(config, model_name, stats),
            )
            extracted.append((model_name, path, stats, sha256))
            del model, tokenizer, tensors, repeated
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        uploaded = []
        for model_name, path, stats, sha256 in extracted:
            remote_path = (
                f"{artifact_config['prefix']}/activations/smoke/"
                f"seed_{seed}/{model_name}.safetensors"
            )
            artifact = upload_bucket_file(path, artifact_config["bucket"], remote_path)
            if artifact.sha256 != sha256:
                raise RuntimeError(f"Local checksum changed before uploading {model_name}.")
            uploaded.append((model_name, stats, artifact))

    _report_smoke_test(tracker, artifact_config, seed, uploaded)


def _extract(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    model_config: dict[str, Any],
):
    activations = config["activations"]
    return extract_activation_tensors(
        rows,
        tokenizer,
        model,
        num_layers=model_config["layers"],
        hidden_size=model_config["hidden_size"],
        normalized_depths=activations["normalized_depths"],
        max_length=activations["max_length"],
        batch_size=activations["batch_size"],
        storage_dtype=getattr(torch, activations["storage_dtype"]),
    )


def _load_splits(config: dict[str, Any]):
    dataset = config["dataset"]
    train_spec = dataset["train"]
    test_spec = dataset["test"]
    train_rows = load_huggingface_dataset(
        dataset["id"],
        dataset["revision"],
        subset=train_spec["subset"],
        split=train_spec["split"],
    )
    test_rows = load_huggingface_dataset(
        dataset["id"],
        dataset["revision"],
        subset=test_spec["subset"],
        split=test_spec["split"],
    )
    sampling = config["sampling"]
    return prepare_splits(
        train_rows,
        test_rows,
        train_size=sampling["train_size"],
        validation_size=sampling["validation_size"],
        seeds=config["data_seeds"],
        prompt_field=dataset["prompt_field"],
        label_field=dataset["label_field"],
        positive_label=dataset["positive_label"],
        negative_label=dataset["negative_label"],
        adversarial_field=dataset["adversarial_field"],
    )


def _activation_metadata(config: dict[str, Any], model_name: str, stats: Any) -> dict[str, str]:
    model = config["models"][model_name]
    activations = config["activations"]
    return {
        "model": model["id"],
        "model_revision": model["revision"],
        "dataset": config["dataset"]["id"],
        "dataset_revision": config["dataset"]["revision"],
        "rows": str(stats.rows),
        "seed": str(config["extraction"]["seed"]),
        "block_indices": json.dumps(stats.block_indices),
        "normalized_depths": json.dumps(activations["normalized_depths"]),
        "max_length": str(activations["max_length"]),
        "token_position": activations["token_position"],
        "activation_boundary": "transformers_output_hidden_states",
    }


def _report_smoke_test(
    tracker: Tracker, artifacts: dict[str, Any], seed: int, uploaded: list[Any]
) -> None:
    lines = [
        f"| {name} | {stats.rows} | {stats.truncation_rate:.2%} | `{artifact.sha256}` |"
        for name, stats, artifact in uploaded
    ]
    tracker.report(
        "Activation smoke test",
        "| Model | Rows | Truncated | SHA-256 |\n"
        "| --- | ---: | ---: | --- |\n"
        + "\n".join(lines)
        + f"\n\nBucket: `hf://buckets/{artifacts['bucket']}/"
        f"{artifacts['prefix']}/activations/smoke/seed_{seed}/`.",
    )


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        "adversarial": dict(sorted(Counter(str(row["adversarial"]) for row in rows).items())),
    }


def _seed_report(seed: int, splits: dict[str, list[dict[str, Any]]]) -> str:
    train = _counts(splits["train"])
    validation = _counts(splits["validation"])
    return (
        f"### Seed {seed}\n\n"
        f"- Train: {train['rows']:,} rows; {train['labels'][0]:,} unharmful and "
        f"{train['labels'][1]:,} harmful.\n"
        f"- Validation: {validation['rows']:,} rows; "
        f"{validation['labels'][0]:,} unharmful and "
        f"{validation['labels'][1]:,} harmful."
    )


def _validate_config(config: dict[str, Any]) -> None:
    hidden_sizes = {model["hidden_size"] for model in config["models"].values()}
    if hidden_sizes != {4096}:
        raise ValueError("All baseline models must expose 4,096-dimensional activations.")
    activations = config["activations"]
    if activations["primary_depth"] not in activations["normalized_depths"]:
        raise ValueError("The primary depth must appear in normalized_depths.")
    sampling = config["sampling"]
    if sampling["train_size"] % 2 or sampling["validation_size"] % 2:
        raise ValueError("Train and validation sizes must be even for label balancing.")
    if (
        sampling.get("balance_labels") is not True
        or sampling.get("stratify_by") != ["adversarial"]
        or sampling.get("protect_test") is not True
    ):
        raise ValueError(
            "The baseline requires label balance, adversarial strata, and test protection."
        )
    if activations["prompt_format"] != "raw" or activations["token_position"] != "last_non_padding":
        raise ValueError("Only raw prompts and last-non-padding activations are implemented.")
    if len(activations["normalized_depths"]) != len(set(activations["normalized_depths"])):
        raise ValueError("Normalized activation depths must be unique.")
    extraction = config["extraction"]
    if extraction["seed"] not in config["data_seeds"]:
        raise ValueError("The extraction seed must be one of data_seeds.")
    if extraction["sample_size"] < 2 or extraction["sample_size"] % 2:
        raise ValueError("The extraction sample size must be positive and even.")
    if not 0 < extraction["repeatability_rows"] <= extraction["sample_size"]:
        raise ValueError("repeatability_rows must lie within the extraction sample.")
    unknown_models = set(extraction["models"]) - set(config["models"])
    if unknown_models:
        raise ValueError(f"Unknown extraction models: {sorted(unknown_models)}")
    if not extraction["models"] or len(extraction["models"]) != len(set(extraction["models"])):
        raise ValueError("Extraction models must be non-empty and unique.")
    if config["artifacts"].get("backend") != "huggingface_bucket":
        raise ValueError("Activation artifacts require the Hugging Face bucket backend.")
