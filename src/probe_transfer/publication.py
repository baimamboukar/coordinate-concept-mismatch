import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from probe_transfer.activations import bucket_uri
from probe_transfer.extraction_job import STAGING_ENV, resolve_model_name

ACTIVATION_FILES = {
    "completion.json",
    "seed_42_train.safetensors",
    "seed_42_validation.safetensors",
    "seed_137_train.safetensors",
    "seed_137_validation.safetensors",
    "test.safetensors",
}
TOKEN_ENVIRONMENTS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def publish_model_activations(config: dict[str, Any], model_name: str | None = None) -> str:
    selected = resolve_model_name(config, model_name)
    staging = os.getenv(STAGING_ENV) or config.get("extraction", {}).get("staging_dir")
    if not staging:
        raise RuntimeError(f"{STAGING_ENV} is required for worker publication.")
    source = Path(staging).expanduser().resolve() / "activations" / selected
    _validate_source(source)

    artifacts = config["artifacts"]
    if (
        artifacts.get("worker_upload") is not True
        or artifacts.get("verify_anonymously") is not True
    ):
        raise ValueError("Activation publication must upload and verify directly on the worker.")
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required only for the worker publication step.")
    hf = shutil.which("hf")
    if hf is None:
        raise RuntimeError("The Hugging Face CLI is required for worker publication.")

    remote = bucket_uri(artifacts["bucket"], f"{artifacts['prefix']}/activations/{selected}")
    subprocess.run(
        [hf, "buckets", "sync", str(source), remote, "--no-delete", "--quiet"],
        check=True,
    )

    anonymous = os.environ.copy()
    for name in TOKEN_ENVIRONMENTS:
        anonymous.pop(name, None)
    anonymous["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    with TemporaryDirectory(prefix=f".{selected}-hf-verify-", dir=source.parent) as temporary:
        downloaded = Path(temporary)
        subprocess.run(
            [hf, "buckets", "sync", remote, str(downloaded), "--no-delete", "--quiet"],
            check=True,
            env=anonymous,
        )
        _assert_same_directory(source, downloaded)
    return remote


def _validate_source(source: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Activation directory not found: {source}")
    files = {path.name for path in source.iterdir() if path.is_file()}
    if files != ACTIVATION_FILES or any(path.is_symlink() for path in source.iterdir()):
        raise ValueError("Worker activation directory does not contain exactly six regular files.")


def _assert_same_directory(source: Path, downloaded: Path) -> None:
    local = {path.name: path for path in source.iterdir() if path.is_file()}
    remote = {path.name: path for path in downloaded.iterdir() if path.is_file()}
    if set(local) != ACTIVATION_FILES or set(remote) != ACTIVATION_FILES:
        raise RuntimeError("Anonymous activation download changed the file set.")
    for name in ACTIVATION_FILES:
        if local[name].stat().st_size != remote[name].stat().st_size:
            raise RuntimeError(f"Published activation size changed: {name}")
        if _sha256(local[name]) != _sha256(remote[name]):
            raise RuntimeError(f"Published activation bytes changed: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
