import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch


def validate_cuda_runtime(execution: dict[str, Any], dtype: str = "bfloat16") -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != execution["gpu_count"]:
        raise RuntimeError("Extraction requires exactly one available CUDA device.")
    name = torch.cuda.get_device_name(0)
    accelerators = execution.get("accelerators") or [execution.get("accelerator")]
    if not any(accelerator and accelerator in name for accelerator in accelerators):
        raise RuntimeError(f"Expected one of {accelerators}, found {name}.")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory_gb < execution["minimum_gpu_memory_gb"]:
        minimum = execution["minimum_gpu_memory_gb"]
        raise RuntimeError(f"GPU memory is {memory_gb:.1f} GiB; expected at least {minimum} GiB.")
    driver_support = _cuda_driver_support_version()
    if driver_support < execution["minimum_cuda_driver_support"]:
        raise RuntimeError(
            f"CUDA driver support {driver_support} does not satisfy the frozen environment."
        )
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16.")
    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported execution dtype: {dtype}")
    probe = torch.ones((16, 16), device="cuda", dtype=torch_dtype)
    if not torch.isfinite(probe @ probe).all():
        raise RuntimeError("The CUDA bfloat16 execution smoke test failed.")
    return {
        "gpu": name,
        "memory_gb": memory_gb,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver_support": driver_support,
    }


def _cuda_driver_support_version() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, check=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Unable to verify CUDA driver support with nvidia-smi.") from error
    match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9]+\.[0-9]+)", result.stdout)
    if match is None:
        raise RuntimeError("nvidia-smi did not report CUDA driver support.")
    return float(match.group(1))


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
