from copy import deepcopy

import torch
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from probe_transfer.symmetry.transforms import (
    permute_gpt_neox_residual,
    permute_mistral_residual,
    permute_residual,
    relative_permutation,
    seeded_permutation,
)


def tiny_neox() -> GPTNeoXForCausalLM:
    config = GPTNeoXConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=37,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    return GPTNeoXForCausalLM(config).eval()


def tiny_mistral() -> MistralForCausalLM:
    config = MistralConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=37,
        attention_dropout=0.0,
    )
    return MistralForCausalLM(config).eval()


def tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=37,
        attention_dropout=0.0,
    )
    return LlamaForCausalLM(config).eval()


def tiny_qwen() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=37,
        attention_dropout=0.0,
    )
    return Qwen3ForCausalLM(config).eval()


def test_residual_permutation_preserves_logits_and_permutes_hidden_states() -> None:
    torch.manual_seed(7)
    reference = tiny_neox()
    transformed = deepcopy(reference)
    permutation = seeded_permutation(16, 42)
    input_ids = torch.randint(0, 37, (3, 9))

    permute_gpt_neox_residual(transformed, permutation)
    with torch.inference_mode():
        expected = reference(input_ids, output_hidden_states=True)
        actual = transformed(input_ids, output_hidden_states=True)

    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
    for expected_hidden, actual_hidden in zip(
        expected.hidden_states, actual.hidden_states, strict=True
    ):
        torch.testing.assert_close(
            actual_hidden,
            expected_hidden.index_select(-1, permutation),
            rtol=1e-5,
            atol=1e-6,
        )


def test_relative_permutation_reaches_second_absolute_basis() -> None:
    torch.manual_seed(11)
    reference = tiny_neox()
    transformed = deepcopy(reference)
    first = seeded_permutation(16, 42)
    second = seeded_permutation(16, 137)
    input_ids = torch.randint(0, 37, (2, 7))

    permute_gpt_neox_residual(transformed, first)
    permute_gpt_neox_residual(transformed, relative_permutation(first, second))
    with torch.inference_mode():
        expected = reference(input_ids, output_hidden_states=True)
        actual = transformed(input_ids, output_hidden_states=True)

    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual.hidden_states[-1],
        expected.hidden_states[-1].index_select(-1, second),
        rtol=1e-5,
        atol=1e-6,
    )


def test_mistral_residual_permutation_preserves_function() -> None:
    torch.manual_seed(17)
    reference = tiny_mistral().double()
    transformed = deepcopy(reference)
    permutation = seeded_permutation(16, 42)
    input_ids = torch.randint(0, 37, (3, 9))

    permute_mistral_residual(transformed, permutation)
    with torch.inference_mode():
        expected = reference(input_ids, output_hidden_states=True)
        actual = transformed(input_ids, output_hidden_states=True)

    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-6, atol=1e-7)
    for expected_hidden, actual_hidden in zip(
        expected.hidden_states, actual.hidden_states, strict=True
    ):
        torch.testing.assert_close(
            actual_hidden,
            expected_hidden.index_select(-1, permutation),
            rtol=1e-6,
            atol=1e-7,
        )


def test_llama_residual_permutation_preserves_function() -> None:
    _assert_llama_family_permutation(tiny_llama())


def test_qwen_residual_permutation_preserves_function() -> None:
    _assert_llama_family_permutation(tiny_qwen())


def _assert_llama_family_permutation(model: torch.nn.Module) -> None:
    torch.manual_seed(23)
    reference = model.double()
    transformed = deepcopy(reference)
    permutation = seeded_permutation(16, 137)
    input_ids = torch.randint(0, 37, (3, 9))

    permute_residual(transformed, permutation)
    with torch.inference_mode():
        expected = reference(input_ids, output_hidden_states=True)
        actual = transformed(input_ids, output_hidden_states=True)

    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-6, atol=1e-7)
    for expected_hidden, actual_hidden in zip(
        expected.hidden_states, actual.hidden_states, strict=True
    ):
        torch.testing.assert_close(
            actual_hidden,
            expected_hidden.index_select(-1, permutation),
            rtol=1e-6,
            atol=1e-7,
        )
