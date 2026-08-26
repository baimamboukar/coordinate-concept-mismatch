import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.constants import HF_TOKEN_ENVIRONMENTS
from probe_transfer.layout import artifact_uri


@dataclass(frozen=True)
class Publication:
    source: Path
    remote_prefix: str


def publish_artifacts(
    config: dict[str, Any], requests: list[Publication], tracker: Any | None
) -> list[str]:
    if not requests:
        return []
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required on the worker for artifact publication.")
    hf = shutil.which("hf")
    if hf is None:
        raise RuntimeError("The Hugging Face CLI is required for artifact publication.")

    published = []
    for request in requests:
        _validate_source(request.source)
        uri = artifact_uri(config, request.remote_prefix)
        subprocess.run(
            [hf, "buckets", "sync", str(request.source), uri, "--no-delete", "--quiet"],
            check=True,
        )
        _verify_anonymously(hf, request.source, uri)
        published.append(uri)
    if tracker is not None:
        tracker.report("Artifacts", "Published directly from the worker: " + ", ".join(published))
    return published


def _verify_anonymously(hf: str, source: Path, uri: str) -> None:
    anonymous = os.environ.copy()
    for name in HF_TOKEN_ENVIRONMENTS:
        anonymous.pop(name, None)
    anonymous["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    result = subprocess.run(
        [
            hf,
            "buckets",
            "sync",
            str(source),
            uri,
            "--delete",
            "--dry-run",
            "--ignore-times",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=anonymous,
    )
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    headers = [record for record in records if record.get("type") == "header"]
    operations = [record for record in records if record.get("type") == "operation"]
    if len(headers) != 1:
        raise RuntimeError(f"Anonymous verification returned no sync summary: {uri}")
    summary = headers[0].get("summary", {})
    unchanged = (
        summary.get("uploads") == 0
        and summary.get("downloads") == 0
        and summary.get("deletes") == 0
        and summary.get("skips") == sum(path.is_file() for path in source.rglob("*"))
        and all(record.get("action") == "skip" for record in operations)
    )
    if not unchanged:
        raise RuntimeError(f"Published artifact tree differs from the worker output: {uri}")


def _validate_source(source: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {source}")
    entries = list(source.rglob("*"))
    if not any(path.is_file() for path in entries):
        raise ValueError(f"Artifact directory is empty: {source}")
    if any(path.is_symlink() for path in entries):
        raise ValueError(f"Artifact directories cannot contain symlinks: {source}")
