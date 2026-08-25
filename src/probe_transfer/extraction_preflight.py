import gc
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from probe_transfer.activations import assert_repeatable, save_activation_file
from probe_transfer.extraction_job import (
    ROWS_ENV,
    STAGING_ENV,
    extract_rows,
    load_prepared_splits,
    prepared_splits,
    resolve_model_name,
)
from probe_transfer.models import load_activation_model
from probe_transfer.runtime import (
    validate_cuda_runtime,
    validate_free_disk,
    validate_loaded_model,
)


def run_extraction_preflight(config: dict[str, Any], tracker: Any) -> None:
    selected = resolve_model_name(config)
    rows_root = _required_directory(ROWS_ENV)
    output_root = _required_directory(STAGING_ENV, create=True)
    prepared = load_prepared_splits(rows_root, prepared_splits(config))
    execution = config["execution"]
    free_gb = validate_free_disk(output_root, execution["minimum_disk_free_gb"])
    runtime = validate_cuda_runtime(execution)

    model_config = config["models"][selected]
    tokenizer, model = load_activation_model(
        model_config["id"],
        model_config["revision"],
        dtype=config["activations"]["dtype"],
    )
    try:
        validate_loaded_model(
            model,
            layers=model_config["layers"],
            hidden_size=model_config["hidden_size"],
        )
        rows = prepared[0][1][: config["extraction"]["repeatability_rows"]]
        tensors, stats = extract_rows(rows, tokenizer, model, model_config, config["activations"])
        repeated, _ = extract_rows(rows, tokenizer, model, model_config, config["activations"])
        assert_repeatable(
            tensors,
            repeated,
            rows=len(rows),
            atol=config["extraction"]["repeatability_atol"],
        )
        with TemporaryDirectory(prefix=f"{selected}-preflight-", dir=output_root) as temporary:
            save_activation_file(
                Path(temporary) / "smoke.safetensors",
                tensors,
                {"model": model_config["id"], "revision": model_config["revision"]},
            )
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    tracker.metrics(
        {
            "preflight/rows": float(stats.rows),
            "preflight/free_disk_gb": free_gb,
            "preflight/gpu_memory_gb": runtime["memory_gb"],
        }
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
