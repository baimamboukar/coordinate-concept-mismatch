import gc
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from probe_transfer.extraction.activations import (
    assert_repeatable,
    prompt_truncation_rate,
    save_activation_file,
)
from probe_transfer.extraction.job import (
    ROWS_ENV,
    STAGING_ENV,
    extract_rows,
    load_prepared_splits,
    prepared_splits,
    resolve_model_name,
)
from probe_transfer.extraction.models import load_activation_model
from probe_transfer.extraction.runtime import (
    validate_cuda_runtime,
    validate_free_disk,
    validate_loaded_model,
)


def run_extraction_preflight(
    config: dict[str, Any], tracker: Any, model_name: str | None = None
) -> None:
    selected = resolve_model_name(config, model_name)
    rows_root = _required_directory(config, ROWS_ENV, "rows_dir")
    output_root = _required_directory(config, STAGING_ENV, "staging_dir", create=True)
    prepared = load_prepared_splits(rows_root, prepared_splits(config))
    execution = config["execution"]
    free_gb = validate_free_disk(output_root, execution["minimum_disk_free_gb"])
    runtime = validate_cuda_runtime(execution, config["activations"]["dtype"])

    model_config = config["models"][selected]
    tokenizer, model = load_activation_model(
        model_config["id"],
        model_config["revision"],
        tokenizer_id=config.get("tokenizer", {}).get("id"),
        tokenizer_revision=config.get("tokenizer", {}).get("revision"),
        dtype=config["activations"]["dtype"],
    )
    try:
        validate_loaded_model(
            model,
            layers=model_config["layers"],
            hidden_size=model_config["hidden_size"],
        )
        activations = config["activations"]
        rates = {
            split.name: prompt_truncation_rate(
                rows,
                tokenizer,
                max_length=activations["max_length"],
                add_special_tokens=activations.get("add_special_tokens", True),
            )
            for split, rows in prepared
        }
        limit = float(activations["max_truncation_rate"])
        if any(rate > limit for rate in rates.values()):
            raise ValueError(f"Full-split prompt truncation exceeds {limit:.2%}: {rates}")
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
            "preflight/max_truncation_rate": max(rates.values()),
        }
    )


def _required_directory(
    config: dict[str, Any], name: str, key: str, *, create: bool = False
) -> Path:
    value = os.getenv(name) or config.get("extraction", {}).get(key)
    if not value:
        raise RuntimeError(f"{name} is required.")
    path = Path(value).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return path
