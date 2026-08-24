import torch

from probe_transfer.function_gate import GateOutputs, _gate_record


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
        torch.arange(2),
        {
            "gate_dtype": "float64",
            "logit_atol": 1e-3,
            "logit_rtol": 1e-4,
            "activation_relative_tolerance": 1e-4,
        },
    )

    assert record["maximum_logit_error"] > 1e-3
    assert record["logit_tolerance_passed"] is True
    assert record["passed"] is True
