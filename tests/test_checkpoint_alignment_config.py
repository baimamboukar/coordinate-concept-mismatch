from pathlib import Path

from core.config import load_config
from experiments.pythia_checkpoint_alignment_recovery import _validate_config


def test_checkpoint_alignment_protocol_is_prespecified() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "pythia_checkpoint_alignment_recovery.yaml")

    _validate_config(config)
    assert config["alignment"]["primary_method"] == "permutation_diagonal"
    assert config["alignment"]["negative_control"] == "shuffled_affine_ridge"
    assert config["evaluation"]["bootstrap_samples"] == 2000
    assert config["evaluation"]["retain_row_level"] == [
        "row_id",
        "label",
        "score",
        "probability",
        "prediction",
    ]
