from core.config import load_config
from core.constants import CONFIGS_DIR
from pipeline.config import materialize_stage


def test_checkpoint_alignment_protocol_is_derived() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "pythia_controls.yaml")
    config = materialize_stage(study, "align")

    assert config["alignment"]["primary_method"] == "permutation_diagonal"
    assert config["alignment"]["negative_control"] == "shuffled_affine_ridge"
    assert config["evaluation"]["bootstrap_samples"] == 2000
    assert config["expected_outputs"] == {
        "metrics_rows": 184,
        "prediction_rows": 312616,
        "recovery_rows": 136,
        "alignment_diagnostic_rows": 96,
    }
