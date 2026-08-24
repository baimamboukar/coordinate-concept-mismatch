import gc
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from probe_transfer.activations import (
    ActivationStats,
    assert_repeatable,
    extract_activation_tensors,
    save_activation_file,
)
from probe_transfer.artifacts import write_json
from probe_transfer.data import load_prepared_rows
from probe_transfer.models import load_activation_model

MODEL_ENV = "EXTRACTION_MODEL"
ROWS_ENV = "ACTIVATION_ROWS_DIR"
STAGING_ENV = "ACTIVATION_STAGING_DIR"
DATA_SEEDS = (42, 137)


@dataclass(frozen=True)
class PreparedSplit:
    name: str
    expected_rows: int
    balanced: bool
    data_seed: int | None = None

    @property
    def input_name(self) -> str:
        return f"{self.name}.jsonl"

    @property
    def output_name(self) -> str:
        return f"{self.name}.safetensors"


@dataclass(frozen=True)
class SplitCompletion:
    split: str
    data_seed: int | None
    path: str
    rows: int
    truncated_rows: int
    truncation_rate: float


@dataclass(frozen=True)
class JobCompletion:
    schema_version: int
    status: str
    model_name: str
    model_id: str
    model_revision: str
    block_indices: tuple[int, ...]
    normalized_depths: tuple[float, ...]
    splits: tuple[SplitCompletion, ...]


ModelLoader = Callable[..., tuple[Any, Any]]


def run_extraction_job(
    config: Mapping[str, Any],
    *,
    model_name: str | None = None,
    rows_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_loader: ModelLoader = load_activation_model,
) -> JobCompletion:
    """Extract all prepared splits for exactly one selected model."""
    selected = resolve_model_name(config, model_name)
    prepared_root = _resolve_path(rows_dir, ROWS_ENV, config, "rows_dir")
    staging_root = _resolve_path(output_dir, STAGING_ENV, config, "staging_dir")
    splits = prepared_splits(config)
    _validate_settings(config, splits)

    activations = config["activations"]
    model_config = config["models"][selected]
    model_output = staging_root / "activations" / selected
    if model_output.exists():
        raise FileExistsError(f"Refusing to overwrite model activations: {model_output}")

    tokenizer, model = model_loader(
        model_config["id"],
        model_config["revision"],
        dtype=activations["dtype"],
    )
    completed: list[SplitCompletion] = []
    block_indices: tuple[int, ...] | None = None
    try:
        model_output.mkdir(parents=True)
        for split in splits:
            input_path = prepared_root / split.input_name
            rows = load_prepared_rows(
                input_path,
                split.expected_rows,
                require_balanced=split.balanced,
            )
            tensors, stats = _extract(rows, tokenizer, model, model_config, activations)
            _assert_truncation(stats, float(activations["max_truncation_rate"]), input_path)
            repeated, _ = _extract(
                rows[: config["extraction"]["repeatability_rows"]],
                tokenizer,
                model,
                model_config,
                activations,
            )
            assert_repeatable(
                tensors,
                repeated,
                rows=config["extraction"]["repeatability_rows"],
                atol=config["extraction"]["repeatability_atol"],
            )
            output_path = model_output / split.output_name
            save_activation_file(
                output_path,
                tensors,
                _metadata(config, model_config, split, stats),
            )
            _assert_saved_alignment(output_path, rows, activations["normalized_depths"])
            block_indices = stats.block_indices
            completed.append(_completion(selected, split, stats))
            del tensors, repeated
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if block_indices is None:
        raise RuntimeError("No activation splits were extracted.")
    completion = JobCompletion(
        schema_version=1,
        status="complete",
        model_name=selected,
        model_id=model_config["id"],
        model_revision=model_config["revision"],
        block_indices=block_indices,
        normalized_depths=tuple(float(depth) for depth in activations["normalized_depths"]),
        splits=tuple(completed),
    )
    write_json(model_output / "completion.json", asdict(completion))
    return completion


def resolve_model_name(config: Mapping[str, Any], explicit: str | None = None) -> str:
    extraction = config.get("extraction", {})
    configured = extraction.get("model")
    environment = os.getenv(MODEL_ENV)
    supplied = {value for value in (explicit, environment, configured) if value}
    if len(supplied) > 1:
        raise ValueError("Explicit, environment, and config model selectors disagree.")
    selected = explicit or environment or configured
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"Select one model with extraction.model or {MODEL_ENV}.")
    if selected not in config.get("models", {}):
        raise ValueError(f"Unknown activation model: {selected}")
    if selected not in extraction.get("models", ()):
        raise ValueError(f"Model {selected} is not enabled by extraction.models.")
    return selected


def prepared_splits(config: Mapping[str, Any]) -> tuple[PreparedSplit, ...]:
    sampling = config["sampling"]
    splits = [PreparedSplit("test", sampling["test_size"], False)]
    for seed in DATA_SEEDS:
        splits.extend(
            (
                PreparedSplit(f"seed_{seed}_train", sampling["train_size"], True, seed),
                PreparedSplit(
                    f"seed_{seed}_validation",
                    sampling["validation_size"],
                    True,
                    seed,
                ),
            )
        )
    return tuple(splits)


def _resolve_path(
    explicit: str | Path | None,
    environment_name: str,
    config: Mapping[str, Any],
    config_name: str,
) -> Path:
    value = explicit or os.getenv(environment_name) or config.get("extraction", {}).get(config_name)
    if not value:
        raise ValueError(f"Provide extraction.{config_name} or {environment_name}.")
    return Path(value).expanduser().resolve()


def _validate_settings(config: Mapping[str, Any], splits: tuple[PreparedSplit, ...]) -> None:
    extraction = config["extraction"]
    activations = config["activations"]
    if config.get("stage") != "modern_baseline" or extraction.get("mode") != "full":
        raise ValueError("Model workers require stage=modern_baseline and extraction.mode=full.")
    if tuple(config.get("data_seeds", ())) != DATA_SEEDS:
        raise ValueError(f"Full extraction requires data_seeds={list(DATA_SEEDS)}.")
    depths = activations["normalized_depths"]
    if len(depths) != 4 or len(set(depths)) != 4:
        raise ValueError("Full extraction requires four unique normalized depths.")
    if activations.get("prompt_format") != "raw" or activations.get("token_position") != (
        "last_non_padding"
    ):
        raise ValueError("Full extraction requires raw prompts and last-non-padding activations.")
    repeat_rows = extraction["repeatability_rows"]
    if repeat_rows < 1 or any(repeat_rows > split.expected_rows for split in splits):
        raise ValueError("repeatability_rows must fit every prepared split.")
    if extraction["repeatability_atol"] < 0:
        raise ValueError("repeatability_atol cannot be negative.")
    rate = activations["max_truncation_rate"]
    if not 0 <= rate <= 1:
        raise ValueError("max_truncation_rate must lie in [0, 1].")


def _extract(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    model_config: Mapping[str, Any],
    activations: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], ActivationStats]:
    storage_dtype = getattr(torch, activations["storage_dtype"], None)
    if storage_dtype is None:
        raise ValueError(f"Unsupported storage dtype: {activations['storage_dtype']}")
    return extract_activation_tensors(
        rows,
        tokenizer,
        model,
        num_layers=model_config["layers"],
        hidden_size=model_config["hidden_size"],
        normalized_depths=activations["normalized_depths"],
        max_length=activations["max_length"],
        batch_size=activations["batch_size"],
        storage_dtype=storage_dtype,
    )


def _assert_truncation(stats: ActivationStats, limit: float, path: Path) -> None:
    if stats.truncation_rate > limit:
        raise ValueError(f"{path} truncation rate {stats.truncation_rate:.2%} exceeds {limit:.2%}.")


def _assert_saved_alignment(
    path: Path, rows: list[dict[str, Any]], normalized_depths: list[float]
) -> None:
    expected_ids = torch.tensor([row["row_id"] for row in rows])
    expected_labels = torch.tensor([row["label"] for row in rows])
    layer_keys = {f"layer_{round(depth * 100)}" for depth in normalized_depths}
    with safe_open(path, framework="pt", device="cpu") as saved:
        if not torch.equal(saved.get_tensor("row_ids"), expected_ids):
            raise RuntimeError(f"Saved row order changed for {path}.")
        if not torch.equal(saved.get_tensor("labels"), expected_labels):
            raise RuntimeError(f"Saved labels changed for {path}.")
        if any(saved.get_slice(key).get_shape()[0] != len(rows) for key in layer_keys):
            raise RuntimeError(f"Saved activation count changed for {path}.")


def _completion(model_name: str, split: PreparedSplit, stats: ActivationStats) -> SplitCompletion:
    relative = Path("activations") / model_name / split.output_name
    return SplitCompletion(
        split.name,
        split.data_seed,
        str(relative),
        stats.rows,
        stats.truncated_rows,
        stats.truncation_rate,
    )


def _metadata(
    config: Mapping[str, Any],
    model: Mapping[str, Any],
    split: PreparedSplit,
    stats: ActivationStats,
) -> dict[str, str]:
    activations = config["activations"]
    dataset = config["dataset"]
    return {
        "model": str(model["id"]),
        "model_revision": str(model["revision"]),
        "dataset": str(dataset["id"]),
        "dataset_revision": str(dataset["revision"]),
        "split": split.name,
        "data_seed": "shared" if split.data_seed is None else str(split.data_seed),
        "rows": str(stats.rows),
        "block_indices": json.dumps(stats.block_indices),
        "normalized_depths": json.dumps(activations["normalized_depths"]),
        "max_length": str(activations["max_length"]),
        "token_position": "last_non_padding",
        "activation_boundary": "transformers_output_hidden_states",
    }
