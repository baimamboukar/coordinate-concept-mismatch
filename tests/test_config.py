from pathlib import Path

import pytest

from core.config import ConfigError, load_config
from core.constants import CONFIGS_DIR
from experiments.frozen_probe_transfer_baseline import _validate_config
from experiments.frozen_probe_transfer_baseline_pythia_pilot import (
    _validate_config as validate_pythia_pilot,
)


def write_config(path: Path, name: str) -> None:
    path.write_text(
        f"name: {name}\n"
        "runner: experiments.frozen_probe_transfer_baseline:run\n"
        "seed: 42\n"
        "evaluation:\n"
        "  primary_metrics: [auroc]\n"
        "  secondary_metrics: [auprc, accuracy, balanced_accuracy, precision, recall, "
        "f1, tn, fp, fn, tp, tpr_at_fpr]\n"
        "  retain_row_level: [row_id, label, score, probability, prediction]\n"
    )


def test_loads_semantically_named_config(tmp_path: Path) -> None:
    path = tmp_path / "frozen_probe_transfer_baseline.yaml"
    write_config(path, path.stem)

    config = load_config(path)

    assert config["name"] == "frozen_probe_transfer_baseline"


def test_rejects_number_only_experiment_name(tmp_path: Path) -> None:
    path = tmp_path / "exp_001.yaml"
    write_config(path, path.stem)

    with pytest.raises(ConfigError, match="Number-only"):
        load_config(path)


def test_requires_filename_to_match_name(tmp_path: Path) -> None:
    path = tmp_path / "frozen_probe_transfer_baseline.yaml"
    write_config(path, "cross_family_probe_transfer")

    with pytest.raises(ConfigError, match="filename"):
        load_config(path)


def test_huggingface_resources_require_commit_revision(tmp_path: Path) -> None:
    path = tmp_path / "frozen_probe_transfer_baseline.yaml"
    path.write_text(
        "name: frozen_probe_transfer_baseline\n"
        "runner: experiments.frozen_probe_transfer_baseline:run\n"
        "seed: 42\n"
        "models:\n"
        "  source:\n"
        "    id: organization/model\n"
        "    revision: main\n"
    )

    with pytest.raises(ConfigError, match="40-character commit"):
        load_config(path)


def test_baseline_config_scopes_modern_pair_and_controls() -> None:
    config = load_config(CONFIGS_DIR / "frozen_probe_transfer_baseline.yaml")

    assert list(config["models"]) == ["llama", "qwen", "nemotron"]
    assert config["models"]["qwen"]["id"] == "Qwen/Qwen3-8B"
    assert {model["hidden_size"] for model in config["models"].values()} == {4096}
    assert config["data_seeds"] == [42, 137]
    assert config["stage"] == "modern_baseline"
    assert config["training"] is True
    assert config["sampling"]["test_size"] == 1699
    assert config["activations"]["normalized_depths"] == [0.25, 0.5, 0.75, 1.0]
    assert config["extraction"]["mode"] == "full"
    assert config["extraction"]["models"] == ["llama", "qwen", "nemotron"]

    jobs = config["extraction"]["jobs"]
    assert {job["model"] for job in jobs} == {"llama", "qwen", "nemotron"}
    assert all(job["accelerator"] == "H100" and job["gpu_count"] == 1 for job in jobs)

    pair_groups = config["evaluation"]["pair_groups"]
    assert pair_groups["primary"] == [["llama", "qwen"], ["qwen", "llama"]]
    assert len(pair_groups["lineage_control"]) == 2
    assert len(pair_groups["exploratory"]) == 2
    assert config["claims"]["broad_three_family"] == "pending"

    assert config["tracking"] == {"wandb": True, "mode": "offline"}
    assert config["artifacts"]["bucket"] == "baimamboukar/coordinate-concept-mismatch"
    assert config["artifacts"]["defer_upload"] is True
    assert config["artifacts"]["worker_upload"] is True
    assert config["artifacts"]["verify_anonymously"] is True
    assert config["evaluation"]["primary_metrics"] == ["auroc", "auroc_transfer_gap"]
    assert config["evaluation"]["operating_fprs"] == [0.01, 0.05]
    _validate_config(config)


def test_cross_family_baseline_extension_is_prespecified() -> None:
    path = CONFIGS_DIR / "frozen_probe_transfer_baseline_cross_family_extension.yaml"
    config = load_config(path)

    _validate_config(config)
    assert list(config["models"]) == ["llama", "qwen", "nemotron", "mistral", "granite"]
    assert config["extraction"]["models"] == ["mistral", "granite"]
    assert config["models"]["mistral"]["layers"] == 32
    assert config["models"]["granite"]["layers"] == 40
    assert len(config["evaluation"]["pair_groups"]["primary"]) == 10
    assert config["expected_outputs"] == {
        "metrics_rows": 300,
        "prediction_rows": 509700,
        "transfer_gap_rows": 240,
        "probe_bundles": 10,
    }
    assert config["artifacts"]["worker_upload"] is True
    assert config["artifacts"]["verify_anonymously"] is True


def test_baseline_pythia_pilot_config_is_pinned_and_valid() -> None:
    config = load_config(CONFIGS_DIR / "frozen_probe_transfer_baseline_pythia_pilot.yaml")

    assert config["name"] == "frozen_probe_transfer_baseline_pythia_pilot"
    assert config["stage"] == "pythia_pilot"
    assert config["data_seeds"] == [42, 137]
    assert list(config["models"]) == ["pythia_seed1234", "pythia_seed1"]
    assert config["models"]["pythia_seed1"]["revision"] == (
        "33803c4f5a1e9a4bece59e52f17bb8755add33e1"
    )
    assert config["tracking"]["wandb"] is True
    assert config["artifacts"]["defer_upload"] is True
    assert config["artifacts"]["prefix"] == (
        "experiments/frozen_probe_transfer_baseline/pythia_pilot"
    )
    validate_pythia_pilot(config)


def test_rejects_experiment_without_primary_and_secondary_metrics(tmp_path: Path) -> None:
    path = tmp_path / "missing_metrics.yaml"
    path.write_text("name: missing_metrics\nrunner: module:function\nseed: 42\n")

    with pytest.raises(ConfigError, match="evaluation mapping"):
        load_config(path)
