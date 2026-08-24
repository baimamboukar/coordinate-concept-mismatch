from typing import Any

import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from probe_transfer.function_gate import _collect_outputs


class FixedTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([[1, 2, 3]]).repeat(len(prompts), 1)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def test_gate_collection_preserves_fp64_outputs() -> None:
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=37,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
    ).double()

    outputs = _collect_outputs(
        FixedTokenizer(),
        model.eval(),
        [{"prompt": "test"}],
        {"layers": 2},
        {"probed_depths": [0.5, 1.0], "gate_batch_size": 1, "gate_max_length": 8},
    )

    assert outputs.logits.dtype == torch.float64
    assert all(hidden.dtype == torch.float64 for hidden in outputs.hidden_states.values())
