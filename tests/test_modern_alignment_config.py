from pathlib import Path

from core.config import load_config
from experiments.modern_activation_alignment_recovery import _validate_config
from probe_transfer.alignment_materials import direction_groups


def test_modern_alignment_protocol_is_prespecified() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "modern_activation_alignment_recovery.yaml")

    _validate_config(config)
    assert direction_groups(config) == [
        ("llama", "qwen", "primary"),
        ("qwen", "llama", "primary"),
        ("llama", "nemotron", "lineage_control"),
        ("nemotron", "llama", "lineage_control"),
        ("qwen", "nemotron", "exploratory"),
        ("nemotron", "qwen", "exploratory"),
    ]
    assert config["alignment"]["primary_probe_family"] == "linear"
    assert config["evaluation"]["bootstrap_samples"] == 2000
    assert config["expected_outputs"]["prediction_rows"] == 448536


def test_two_model_alignment_retains_bidirectional_primary_default() -> None:
    config = {"models": {"first": {}, "second": {}}, "evaluation": {}}

    assert direction_groups(config) == [
        ("first", "second", "primary"),
        ("second", "first", "primary"),
    ]
