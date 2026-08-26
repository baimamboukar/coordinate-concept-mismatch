import numpy as np
import pytest

from probe_transfer.probes.training import families_for_depth, fit_probe_family


def test_depth_policy_keeps_nonlinear_probes_at_primary_depth() -> None:
    config = {
        "primary_families": ["linear", "cp_degree_2", "mlp"],
        "secondary_families": ["linear"],
    }

    assert families_for_depth(config, 0.75, 0.75) == ["linear", "cp_degree_2", "mlp"]
    assert families_for_depth(config, 0.5, 0.75) == ["linear"]


def test_linear_family_returns_selected_hyperparameter() -> None:
    train_x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    train_y = np.array([0, 0, 1, 1])

    selected = fit_probe_family(
        "linear",
        train_x,
        train_y,
        train_x,
        train_y,
        {"linear": {"c_values": [0.1, 1.0], "max_iter": 100}},
        device="cpu",
    )

    assert selected.parameters["c"] == 0.1


def test_unknown_probe_family_fails() -> None:
    values = np.array([[-1.0], [1.0]])
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="Unsupported"):
        fit_probe_family("cubic", values, labels, values, labels, {}, device="cpu")
