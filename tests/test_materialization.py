import subprocess
from pathlib import Path

from probe_transfer.materialization import (
    materialize_baseline,
    materialize_fit_activations,
    materialize_recovery_reference,
)


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


def test_site_specific_baseline_uses_variant_paths(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("probe_transfer.materialization.shutil.which", lambda _name: "/usr/bin/hf")
    monkeypatch.setattr(
        "probe_transfer.materialization.subprocess.run",
        lambda command, **kwargs: calls.append(command),
    )
    config = {
        "artifacts": {"bucket": "test/project", "dataset_key": "wildguardmix-v1"},
        "activations": {"artifact_key": "mlp-intermediate"},
        "data_seeds": [42, 137],
        "materials": {
            "source_name": "modern_mlp_neuron_permutation_probe_transport",
            "source_study": "modern_mlp_neuron_symmetry",
            "source_variant": "baseline",
        },
        "models": {"mistral": {"artifact_key": "mistral-7b-v0.3"}},
    }

    materialize_baseline(config, tmp_path, models=["mistral"])

    commands = [" ".join(command) for command in calls]
    assert any("modern-mlp-neuron-symmetry/baseline/probes" in command for command in commands)
    assert any("mistral-7b-v0.3/mlp-intermediate" in command for command in commands)


def test_cross_task_fit_uses_the_configured_activation_dataset(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("probe_transfer.materialization.shutil.which", lambda _name: "/usr/bin/hf")
    monkeypatch.setattr(
        "probe_transfer.materialization.subprocess.run",
        lambda command, **kwargs: calls.append(command),
    )
    config = {
        "artifacts": {"bucket": "test/project", "dataset_key": "wildguardmix-v1"},
        "fit_materials": {"dataset_key": "sst2-sentiment-v1"},
        "models": {
            "ai2": {"artifact_key": "ai2-olmo-1b"},
            "amd": {"artifact_key": "amd-olmo-1b"},
        },
    }

    materialize_fit_activations(config, tmp_path)

    commands = [" ".join(command) for command in calls]
    assert len(commands) == 2
    assert all("activations/sst2-sentiment-v1" in command for command in commands)
    assert all("wildguardmix-v1" not in command for command in commands)


def test_cross_task_reference_materializes_only_recovery_rows(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("probe_transfer.materialization.shutil.which", lambda _name: "/usr/bin/hf")
    monkeypatch.setattr(
        "probe_transfer.materialization.subprocess.run",
        lambda command, **kwargs: calls.append(command),
    )
    config = {
        "artifacts": {"bucket": "test/project"},
        "reference_materials": {
            "source_name": "olmo1_independent_alignment",
            "source_study": "olmo1_independent_training_wildguard",
            "source_variant": "wildguard-alignment",
        },
    }

    path = materialize_recovery_reference(config, tmp_path)

    assert path == tmp_path / "results" / "recovery.jsonl"
    assert len(calls) == 1
    command = calls[0]
    assert "recovery.jsonl" in command
    assert "predictions.jsonl" not in command
