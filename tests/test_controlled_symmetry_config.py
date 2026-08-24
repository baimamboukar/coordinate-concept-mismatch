from core.config import load_config
from core.constants import CONFIGS_DIR
from experiments.pythia_residual_permutation_probe_transport import _validate_config


def test_controlled_symmetry_config_is_pinned_and_valid() -> None:
    config = load_config(CONFIGS_DIR / "pythia_residual_permutation_probe_transport.yaml")

    assert config["stage"] == "controlled_residual_permutation"
    assert config["training"] is False
    assert config["symmetry"]["permutation_seeds"] == [42, 137]
    assert config["materials"]["expected_test_rows"] == 1699
    assert config["symmetry"]["gate_rows"] == 1699
    assert config["evaluation"]["primary_metrics"] == [
        "raw_auroc_gap",
        "recovery_fraction",
    ]
    assert config["artifacts"]["prefix"] == (
        "experiments/pythia_residual_permutation_probe_transport"
    )
    _validate_config(config)
