import pytest
import torch

from probe_transfer.symmetry.coordinates import CoordinateTransform
from probe_transfer.symmetry.gate import GateOutputs, _gate_record
from probe_transfer.symmetry.runner import _require_gate_pass


def test_function_gate_uses_combined_logit_tolerance() -> None:
    reference = GateOutputs(
        logits=torch.tensor([[10.0, 0.0]]),
        hidden_states={"layer_75": torch.tensor([[1.0, 2.0]])},
    )
    actual = GateOutputs(
        logits=torch.tensor([[10.0015, 0.0005]]),
        hidden_states={"layer_75": torch.tensor([[1.0, 2.0]])},
    )

    record = _gate_record(
        "model",
        "permutation",
        42,
        reference,
        actual,
        CoordinateTransform.identity("permutation", 2),
        {
            "transformation": "residual_permutation",
            "gate_dtype": "float64",
            "logit_atol": 1e-3,
            "logit_rtol": 1e-4,
            "activation_relative_tolerance": 1e-4,
        },
    )

    assert record["maximum_logit_error"] > 1e-3
    assert record["logit_tolerance_passed"] is True
    assert record["passed"] is True


def test_failed_function_gate_is_preserved(tmp_path) -> None:
    gate = {
        "transformation_seed": 137,
        "maximum_logit_error": 2e-5,
        "maximum_activation_relative_error": 3e-6,
        "next_token_agreement": 1.0,
        "passed": False,
    }

    with pytest.raises(RuntimeError, match="seed=137 logits=2.000e-05"):
        _require_gate_pass([gate], "full-test", tmp_path)

    assert (tmp_path / "diagnostics" / "full-test_function_gates.jsonl").is_file()
