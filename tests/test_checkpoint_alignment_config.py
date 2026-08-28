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


def test_boolq_protocol_uses_composed_prompts_and_compact_outputs() -> None:
    study = load_config(CONFIGS_DIR / "studies" / "olmo1_independent_training_boolq.yaml")
    transfer = materialize_stage(study, "transfer")
    alignment = materialize_stage(study, "align")

    assert study["dataset"]["prompt_template"] == "Question: {question}\nPassage: {passage}"
    assert study["sampling"] == {
        "train_size": 6000,
        "validation_size": 1000,
        "test_size": 3270,
        "balance_labels": True,
        "stratify_by": [],
        "protect_test": True,
    }
    assert transfer["expected_outputs"] == {
        "metrics_rows": 24,
        "prediction_rows": 78480,
        "transfer_gap_rows": 12,
        "probe_bundles": 4,
    }
    assert alignment["alignment"]["methods"] == ["affine_ridge", "orthogonal_procrustes"]
    assert alignment["expected_outputs"] == {
        "metrics_rows": 60,
        "prediction_rows": 196200,
        "recovery_rows": 36,
        "alignment_diagnostic_rows": 12,
    }


def test_heldout_boolq_fits_have_equal_total_budgets() -> None:
    names = (
        "olmo1_map_generalization_sst2_to_boolq",
        "olmo1_map_generalization_wildguard_to_boolq",
        "olmo1_map_generalization_pooled_to_boolq",
    )
    configs = [
        materialize_stage(load_config(CONFIGS_DIR / "studies" / f"{name}.yaml"), "align")
        for name in names
    ]

    assert all(config["fit_materials"]["expected_train_rows"] == 6000 for config in configs)
    assert all(config["artifacts"]["dataset_key"] == "boolq-qa-v1" for config in configs)
    assert all(
        config["reference_materials"]["source_variant"] == "boolq-alignment" for config in configs
    )
    assert [
        [entry["fit_rows"] for entry in config["fit_materials"]["datasets"]] for config in configs
    ] == [[6000], [6000], [3000, 3000]]
