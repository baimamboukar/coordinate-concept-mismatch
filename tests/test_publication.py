import shutil
import subprocess
from pathlib import Path

import pytest

from probe_transfer import publication


def _config(root: Path) -> dict:
    return {
        "models": {"mistral": {}},
        "extraction": {"models": ["mistral"], "staging_dir": str(root)},
        "artifacts": {
            "bucket": "test/project",
            "prefix": "experiments/baseline",
            "worker_upload": True,
            "verify_anonymously": True,
        },
    }


def _activation_source(root: Path) -> Path:
    source = root / "activations" / "mistral"
    source.mkdir(parents=True)
    for index, name in enumerate(sorted(publication.ACTIVATION_FILES)):
        (source / name).write_bytes(f"content-{index}".encode())
    return source


def test_worker_upload_is_anonymously_downloaded_and_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _activation_source(tmp_path)
    calls = []
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(publication.shutil, "which", lambda _name: "/usr/bin/hf")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[3].startswith("hf://"):
            destination = Path(command[4])
            for path in source.iterdir():
                shutil.copy2(path, destination / path.name)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(publication.subprocess, "run", fake_run)

    uri = publication.publish_model_activations(_config(tmp_path), "mistral")

    assert uri == "hf://buckets/test/project/experiments/baseline/activations/mistral"
    assert calls[0][0][3] == str(source)
    assert calls[1][0][3] == uri
    assert "HF_TOKEN" not in calls[1][1]["env"]
    assert calls[1][1]["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_worker_upload_requires_ephemeral_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _activation_source(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        publication.publish_model_activations(_config(tmp_path), "mistral")
