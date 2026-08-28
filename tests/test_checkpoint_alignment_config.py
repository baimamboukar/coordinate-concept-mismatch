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


def test_pooled_map_compatibility_varies_budget_and_evaluation_task() -> None:
    cases = {
        "olmo1_pooled_map_compatibility_equal_sst2": (12000, [6000, 6000], 52320),
        "olmo1_pooled_map_compatibility_equal_wildguard": (12000, [6000, 6000], 101940),
        "olmo1_pooled_map_compatibility_full_sst2": (24000, [12000, 12000], 52320),
        "olmo1_pooled_map_compatibility_full_wildguard": (
            24000,
            [12000, 12000],
            101940,
        ),
    }

    for name, (total, rows, predictions) in cases.items():
        config = materialize_stage(load_config(CONFIGS_DIR / "studies" / f"{name}.yaml"), "align")

        assert config["fit_materials"]["evaluation_included"] is True
        assert config["fit_materials"]["expected_train_rows"] == total
        assert [entry["fit_rows"] for entry in config["fit_materials"]["datasets"]] == rows
        assert config["alignment"]["methods"] == ["affine_ridge", "orthogonal_procrustes"]
        assert config["expected_outputs"] == {
            "metrics_rows": 60,
            "prediction_rows": predictions,
            "recovery_rows": 36,
            "alignment_diagnostic_rows": 12,
        }


def test_heldout_panel_has_pinned_binary_tasks_and_compact_contracts() -> None:
    cases = {
        "olmo1_independent_training_ag_news": (3800, 91200, 228000),
        "olmo1_independent_training_mnli": (6692, 160608, 401520),
    }

    for name, (test_rows, transfer_predictions, alignment_predictions) in cases.items():
        study = load_config(CONFIGS_DIR / "studies" / f"{name}.yaml")
        transfer = materialize_stage(study, "transfer")
        alignment = materialize_stage(study, "align")

        assert study["sampling"]["test_size"] == test_rows
        assert len(study["dataset"]["revision"]) == 40
        assert study["activations"]["normalized_depths"] == [0.75]
        assert transfer["expected_outputs"] == {
            "metrics_rows": 24,
            "prediction_rows": transfer_predictions,
            "transfer_gap_rows": 12,
            "probe_bundles": 4,
        }
        assert alignment["expected_outputs"] == {
            "metrics_rows": 60,
            "prediction_rows": alignment_predictions,
            "recovery_rows": 36,
            "alignment_diagnostic_rows": 12,
        }


def test_heldout_panel_map_conditions_fix_coverage_and_budget() -> None:
    tasks = {
        "ag_news": ("ag-news-world-business-v1", 228000),
        "mnli": ("mnli-entailment-contradiction-v1", 401520),
    }
    fits = {
        "sst2": (12000, [12000]),
        "wildguard": (12000, [12000]),
        "pooled_equal": (12000, [6000, 6000]),
        "pooled_full": (24000, [12000, 12000]),
    }

    for task, (dataset_key, predictions) in tasks.items():
        for fit, (total, rows) in fits.items():
            name = f"olmo1_map_generalization_{fit}_to_{task}"
            config = materialize_stage(
                load_config(CONFIGS_DIR / "studies" / f"{name}.yaml"), "align"
            )

            assert config["artifacts"]["dataset_key"] == dataset_key
            assert config["fit_materials"]["expected_train_rows"] == total
            assert [entry["fit_rows"] for entry in config["fit_materials"]["datasets"]] == rows
            assert config["expected_outputs"]["prediction_rows"] == predictions
