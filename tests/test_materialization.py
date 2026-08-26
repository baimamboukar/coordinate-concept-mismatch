import subprocess
from pathlib import Path

from probe_transfer.materialization import materialize_baseline


def test_alignment_materializes_only_required_baseline_results(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr("probe_transfer.materialization.shutil.which", lambda _name: "/usr/bin/hf")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("probe_transfer.materialization.subprocess.run", fake_run)
    config = {
        "artifacts": {"bucket": "test/project", "dataset_key": "wildguardmix-v1"},
        "materials": {
            "source_name": "frozen_probe_transfer_baseline",
            "source_study": "modern_models",
        },
        "models": {
            "llama": {"artifact_key": "llama-3.1-8b-instruct"},
            "qwen": {"artifact_key": "qwen3-8b"},
        },
    }

    materialize_baseline(config, tmp_path)

    assert len(calls) == 4
    result_call = calls[1]
    assert "--include" in result_call[0]
    assert "metrics.jsonl" in result_call[0]
    assert "HF_TOKEN" not in result_call[1]["env"]
    assert all("predictions.jsonl" not in " ".join(command) for command, _ in calls)


def test_symmetry_materializes_only_selected_model(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("probe_transfer.materialization.shutil.which", lambda _name: "/usr/bin/hf")
    monkeypatch.setattr(
        "probe_transfer.materialization.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    config = {
        "artifacts": {"bucket": "test/project", "dataset_key": "wildguardmix-v1"},
        "data_seeds": [42, 137],
        "materials": {
            "source_name": "frozen_probe_transfer_baseline",
            "source_study": "modern_models",
        },
        "models": {
            "llama": {"artifact_key": "llama-3.1-8b-instruct"},
            "mistral": {"artifact_key": "mistral-7b-v0.3"},
        },
    }

    materialize_baseline(config, tmp_path, models=["mistral"])

    probe_command = calls[0][0]
    assert probe_command.count("--include") == 2
    assert "seed_42/mistral.safetensors" in probe_command
    assert "seed_137/mistral.safetensors" in probe_command
    assert all("llama-3.1-8b-instruct" not in " ".join(command) for command, _ in calls)
    assert any("mistral-7b-v0.3" in " ".join(command) for command, _ in calls)
