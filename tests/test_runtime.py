from types import SimpleNamespace

import pytest

from probe_transfer import runtime


class _Probe:
    def __matmul__(self, _other):
        return self

    def all(self) -> bool:
        return True


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(_index: int) -> str:
        return "NVIDIA H100 PCIe"

    @staticmethod
    def get_device_properties(_index: int):
        return SimpleNamespace(total_memory=80 * 1024**3)

    @staticmethod
    def is_bf16_supported() -> bool:
        return True


def test_runtime_checks_driver_support_not_toolkit_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        cuda=_Cuda(),
        version=SimpleNamespace(cuda="12.8"),
        bfloat16=object(),
        ones=lambda *_args, **_kwargs: _Probe(),
        isfinite=lambda value: value,
    )
    monkeypatch.setattr(runtime, "torch", fake_torch)
    monkeypatch.setattr(runtime, "_cuda_driver_support_version", lambda: 13.0)

    result = runtime.validate_cuda_runtime(
        {
            "accelerator": "H100",
            "gpu_count": 1,
            "minimum_gpu_memory_gb": 75,
            "minimum_cuda_driver_support": 13.0,
        }
    )

    assert result["cuda_runtime"] == "12.8"
    assert result["cuda_driver_support"] == 13.0


def test_driver_support_is_read_from_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(stdout="NVIDIA-SMI 580.00    CUDA Version: 13.0")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_args, **_kwargs: result)

    assert runtime._cuda_driver_support_version() == 13.0
