import shutil
from pathlib import Path
from typing import Any

import torch


def validate_cuda_runtime(execution: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != execution["gpu_count"]:
        raise RuntimeError("Extraction requires exactly one available CUDA device.")
    name = torch.cuda.get_device_name(0)
    if execution["accelerator"] not in name:
        raise RuntimeError(f"Expected {execution['accelerator']}, found {name}.")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory_gb < execution["minimum_gpu_memory_gb"]:
        raise RuntimeError(f"GPU memory is {memory_gb:.1f} GiB; expected at least 75 GiB.")
    runtime = torch.version.cuda
    if (
        runtime is None
        or float(".".join(runtime.split(".")[:2])) < execution["minimum_cuda_driver_support"]
    ):
        raise RuntimeError(f"CUDA runtime {runtime!r} does not satisfy the frozen environment.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16.")
    probe = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite(probe @ probe).all():
        raise RuntimeError("The CUDA bfloat16 execution smoke test failed.")
    return {"gpu": name, "memory_gb": memory_gb, "cuda_runtime": runtime}


def validate_free_disk(path: Path, minimum_gb: float) -> float:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 1024**3
    if free_gb < minimum_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GiB is free; expected at least {minimum_gb} GiB.")
    return free_gb


def validate_loaded_model(model: Any, *, layers: int, hidden_size: int) -> None:
    config = model.config
    if config.num_hidden_layers != layers or config.hidden_size != hidden_size:
        raise RuntimeError(
            "Loaded checkpoint architecture does not match the pinned experiment contract."
        )
    parameter_devices = {parameter.device.type for parameter in model.parameters()}
    if parameter_devices != {"cuda"}:
        raise RuntimeError(f"Model parameters must remain entirely on CUDA: {parameter_devices}")
    device_map = getattr(model, "hf_device_map", {})
    if any(value in {"cpu", "disk"} for value in device_map.values()):
        raise RuntimeError("Automatic placement offloaded part of the model.")
