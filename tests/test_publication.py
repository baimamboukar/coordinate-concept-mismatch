import json
import subprocess
from pathlib import Path

import pytest

from probe_transfer.publication import Publication, publish_artifacts


class Tracker:
    def __init__(self) -> None:
        self.reports = []

    def report(self, heading, body) -> None:
        self.reports.append((heading, body))


def test_worker_upload_is_anonymously_verified_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "results"
    source.mkdir()
    (source / "metrics.jsonl").write_text("{}\n")
    calls = []
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr("probe_transfer.publication.shutil.which", lambda _name: "/usr/bin/hf")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = ""
        if "--dry-run" in command:
            output = json.dumps(
                {
                    "type": "header",
                    "summary": {
                        "uploads": 0,
                        "downloads": 0,
                        "deletes": 0,
                        "skips": 1,
                    },
                }
            )
            output += "\n" + json.dumps(
                {"type": "operation", "action": "skip", "path": "metrics.jsonl"}
            )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr("probe_transfer.publication.subprocess.run", fake_run)
    tracker = Tracker()
    config = {"artifacts": {"bucket": "test/project"}}

    uris = publish_artifacts(
        config,
        [Publication(source, "studies/probe-transfer/modern-models/results")],
        tracker,
    )

    assert uris == ["hf://buckets/test/project/studies/probe-transfer/modern-models/results"]
    assert calls[0][0][3] == str(source)
    assert "--dry-run" in calls[1][0]
    assert "HF_TOKEN" not in calls[1][1]["env"]
    assert calls[1][1]["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert len(tracker.reports) == 1


def test_worker_upload_requires_ephemeral_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "results"
    source.mkdir()
    (source / "metrics.jsonl").write_text("{}\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        publish_artifacts(
            {"artifacts": {"bucket": "test/project"}},
            [Publication(source, "studies/probe-transfer/results")],
            Tracker(),
        )
