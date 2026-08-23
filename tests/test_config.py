from pathlib import Path

import pytest

from core.config import ConfigError, load_config
from core.constants import CONFIGS_DIR
from experiments.frozen_probe_transfer_baseline import _validate_config


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


def test_baseline_config_uses_qwen_and_pinned_resources() -> None:
    config = load_config(CONFIGS_DIR / "frozen_probe_transfer_baseline.yaml")

    assert config["models"]["qwen"]["id"] == "Qwen/Qwen3-8B"
    assert "glm" not in config["models"]
    assert {model["hidden_size"] for model in config["models"].values()} == {4096}
    assert config["data_seeds"] == [42, 137]
    assert config["stage"] == "extract_activations"
    assert config["extraction"]["sample_size"] == 100
    assert config["artifacts"]["bucket"] == "baimamboukar/coordinate-concept-mismatch"
    _validate_config(config)


def test_pythia_smoke_config_is_pinned_and_valid() -> None:
    config = load_config(CONFIGS_DIR / "pythia_activation_smoke.yaml")

    model = config["models"]["pythia_410m"]
    assert model["id"] == "EleutherAI/pythia-410m"
    assert model["revision"] == "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
    assert (model["layers"], model["hidden_size"]) == (24, 1024)
    assert config["extraction"]["models"] == ["pythia_410m"]
    _validate_config(config)


def test_rejects_experiment_without_primary_and_secondary_metrics(tmp_path: Path) -> None:
    path = tmp_path / "missing_metrics.yaml"
    path.write_text("name: missing_metrics\nrunner: module:function\nseed: 42\n")

    with pytest.raises(ConfigError, match="evaluation mapping"):
        load_config(path)
