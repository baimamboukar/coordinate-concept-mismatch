from copy import deepcopy
from typing import Any

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

from probe_transfer.symmetry.transforms import (
    permute_attention_heads,
    relative_permutation,
    seeded_gqa_head_permutation,
    seeded_permutation,
)


def tiny_mistral() -> MistralForCausalLM:
    return (
        MistralForCausalLM(
            MistralConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                vocab_size=37,
                attention_dropout=0.0,
            )
        )
        .double()
        .eval()
    )


def test_seeded_gqa_permutation_preserves_group_associations() -> None:
    permutation = seeded_gqa_head_permutation(8, 2, 4, 42)
    heads = permutation.reshape(8, 4)[:, 0] // 4
    groups = heads.reshape(2, 4) // 4

    assert permutation.shape == (32,)
    assert torch.equal(torch.sort(permutation).values, torch.arange(32))
    assert torch.all(groups == groups[:, :1])
    assert torch.equal(torch.sort(groups[:, 0]).values, torch.arange(2))


def test_attention_head_permutation_preserves_function_and_permutes_site() -> None:
    torch.manual_seed(31)
    reference = tiny_mistral()
    transformed = deepcopy(reference)
    first = seeded_gqa_head_permutation(4, 2, 4, 42)
    second = seeded_gqa_head_permutation(4, 2, 4, 137)
    input_ids = torch.randint(0, 37, (3, 9))

    expected, expected_site = _attention_outputs(reference, input_ids, block_index=2)
    permute_attention_heads(transformed, first, (2,))
    permute_attention_heads(transformed, relative_permutation(first, second), (2,))
    actual, actual_site = _attention_outputs(transformed, input_ids, block_index=2)

    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        actual_site,
        expected_site.index_select(-1, second),
        rtol=1e-6,
        atol=1e-7,
    )
    for expected_hidden, actual_hidden in zip(
        expected.hidden_states, actual.hidden_states, strict=True
    ):
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-6, atol=1e-7)


def test_attention_head_permutation_rejects_unstructured_coordinates() -> None:
    model = tiny_mistral()
    with pytest.raises(ValueError, match="preserve coordinates within each head"):
        permute_attention_heads(model, seeded_permutation(16, 42), (2,))


def _attention_outputs(
    model: MistralForCausalLM, input_ids: torch.Tensor, block_index: int
) -> tuple[Any, torch.Tensor]:
    captured = []
    attention: Any = model.model.layers[block_index - 1].self_attn
    handle = attention.o_proj.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0])
    )
    try:
        with torch.inference_mode():
            output = model(input_ids, output_hidden_states=True)
    finally:
        handle.remove()
    assert len(captured) == 1
    return output, captured[0]
