import gc
import json
import os
from pathlib import Path
from typing import Any

import torch

from core.tracking import Tracker
from probe_transfer.activations import (
    assert_repeatable,
    extract_activation_tensors,
    save_activation_file,
)
from probe_transfer.data import load_prepared_rows
from probe_transfer.models import load_activation_model
from probe_transfer.transfer import run_transfer


def run(config: dict[str, Any], tracker: Tracker) -> None:
    _validate_config(config)
    prepared_dir = _required_directory("PREPARED_DATA_DIR")
    output_dir = _required_directory("ACTIVATION_STAGING_DIR", create=True)
    for name in ("activations", "probes", "results"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir / name}")

    extraction_rows, activation_checksums = _extract_all(config, prepared_dir, output_dir, tracker)
    gaps, result_checksums = run_transfer(output_dir, config, tracker)
    _report_results(tracker, config, gaps)
    _report_artifacts(tracker, {**activation_checksums, **result_checksums})
    tracker.report(
        "Completion",
        f"Extracted {extraction_rows:,} model-row pairs and completed probe training. "
        "Artifacts remain staged for retrieval and verified Hugging Face upload.",
    )


def _extract_all(
    config: dict[str, Any],
    prepared_dir: Path,
    output_dir: Path,
    tracker: Tracker,
) -> tuple[int, dict[str, str]]:
    specs = _split_specs(config)
    activation_config = config["activations"]
    extraction_config = config["extraction"]
    checksums = {}
    rows_processed = 0
    report_rows = []

    for model_name in extraction_config["models"]:
        model_config = config["models"][model_name]
        tokenizer, model = load_activation_model(
            model_config["id"],
            model_config["revision"],
            dtype=activation_config["dtype"],
        )
        repeated = False
        for split_name, expected_size, balanced in specs:
            rows = load_prepared_rows(
                prepared_dir / f"{split_name}.jsonl",
                expected_size,
                require_balanced=balanced,
            )
            tensors, stats = _extract(config, rows, tokenizer, model, model_config)
            if stats.truncation_rate > activation_config["max_truncation_rate"]:
                raise ValueError(
                    f"{model_name}/{split_name} truncation rate {stats.truncation_rate:.2%} "
                    f"exceeds {activation_config['max_truncation_rate']:.2%}."
                )
            if not repeated:
                repeat_rows = extraction_config["repeatability_rows"]
                second, _ = _extract(config, rows[:repeat_rows], tokenizer, model, model_config)
                assert_repeatable(
                    tensors,
                    second,
                    rows=repeat_rows,
                    atol=extraction_config["repeatability_atol"],
                )
                del second
                repeated = True

            relative = f"activations/{model_name}/{split_name}.safetensors"
            checksum = save_activation_file(
                output_dir / relative,
                tensors,
                _activation_metadata(config, model_name, split_name, stats),
            )
            checksums[relative] = checksum
            rows_processed += stats.rows
            report_rows.append(
                f"| {model_name} | {split_name} | {stats.rows:,} | "
                f"{stats.truncation_rate:.2%} | `{checksum}` |"
            )
            del tensors, rows

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tracker.report(
        "Activation extraction",
        "| Model | Split | Rows | Truncated | SHA-256 |\n"
        "| --- | --- | ---: | ---: | --- |\n" + "\n".join(report_rows),
    )
    return rows_processed, checksums


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


def _split_specs(config: dict[str, Any]) -> list[tuple[str, int, bool]]:
    sampling = config["sampling"]
    specs = [("test", sampling["test_size"], False)]
    for seed in config["data_seeds"]:
        specs.extend(
            [
                (f"seed_{seed}_train", sampling["train_size"], True),
                (f"seed_{seed}_validation", sampling["validation_size"], True),
            ]
        )
    return specs


def _activation_metadata(
    config: dict[str, Any], model_name: str, split: str, stats: Any
) -> dict[str, str]:
    model = config["models"][model_name]
    activations = config["activations"]
    return {
        "model": model["id"],
        "model_revision": model["revision"],
        "dataset": config["dataset"]["id"],
        "dataset_revision": config["dataset"]["revision"],
        "split": split,
        "rows": str(stats.rows),
        "block_indices": json.dumps(stats.block_indices),
        "normalized_depths": json.dumps(activations["normalized_depths"]),
        "max_length": str(activations["max_length"]),
        "token_position": activations["token_position"],
        "activation_boundary": "transformers_output_hidden_states",
    }


def _report_results(tracker: Tracker, config: dict[str, Any], gaps: list[dict[str, Any]]) -> None:
    primary_depth = config["activations"]["primary_depth"]
    rows = []
    for gap in gaps:
        if gap["depth"] != primary_depth:
            continue
        rows.append(
            f"| {gap['data_seed']} | {gap['source_model']} → {gap['target_model']} | "
            f"{gap['probe_family']} | {gap['target_oracle_auroc']:.3f} | "
            f"{gap['transfer_auroc']:.3f} | {gap['auroc_gap']:.3f} | "
            f"[{gap['ci_lower']:.3f}, {gap['ci_upper']:.3f}] | "
            f"{'yes' if gap['transfer_failed'] else 'no'} |"
        )
    tracker.report(
        "Primary transfer results",
        "| Data seed | Direction | Probe | Target oracle AUROC | Transfer AUROC | Gap | "
        "95% CI | Prespecified failure |\n"
        "| ---: | --- | --- | ---: | ---: | ---: | --- | --- |\n" + "\n".join(rows),
    )


def _report_artifacts(tracker: Tracker, checksums: dict[str, str]) -> None:
    rows = [f"| `{path}` | `{digest}` |" for path, digest in sorted(checksums.items())]
    tracker.report(
        "Staged artifacts",
        "| File | SHA-256 |\n| --- | --- |\n" + "\n".join(rows),
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
    if config.get("stage") != "full_pilot" or config.get("training") is not True:
        raise ValueError("The Pythia transfer runner requires the full_pilot training stage.")
    if len(config["models"]) != 2 or len(config["data_seeds"]) != 2:
        raise ValueError("The pilot requires exactly two model checkpoints and two data seeds.")
    widths = {model["hidden_size"] for model in config["models"].values()}
    layers = {model["layers"] for model in config["models"].values()}
    if len(widths) != 1 or len(layers) != 1:
        raise ValueError("Pythia restart checkpoints must share width and depth.")
    if config["extraction"]["models"] != list(config["models"]):
        raise ValueError("Every configured model must be extracted in declaration order.")
    sampling = config["sampling"]
    if sampling["train_size"] % 2 or sampling["validation_size"] % 2:
        raise ValueError("Train and validation sizes must be even.")
    activations = config["activations"]
    if activations["primary_depth"] not in activations["normalized_depths"]:
        raise ValueError("The primary depth must be extracted.")
    if activations["token_position"] != "last_non_padding":
        raise ValueError("Only last-non-padding extraction is implemented.")
    if config["artifacts"].get("defer_upload") is not True:
        raise ValueError("The remote worker must stage artifacts without Hugging Face credentials.")
