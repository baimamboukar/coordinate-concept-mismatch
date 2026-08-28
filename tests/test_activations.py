from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import MistralConfig, MistralForCausalLM

from probe_transfer.extraction.activations import (
    assert_repeatable,
    extract_activation_tensors,
    load_activation_split,
    save_activation_file,
)
from probe_transfer.layout import bucket_uri


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(
        self,
        prompts,
        *,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_length=False,
    ):
        token_ids = [list(range(1, len(prompt.split()) + 1)) for prompt in prompts]
        lengths = [len(tokens) for tokens in token_ids]
        if truncation:
            token_ids = [tokens[:max_length] for tokens in token_ids]
        if return_tensors == "pt":
            width = max(len(tokens) for tokens in token_ids)
            input_ids = [tokens + [0] * (width - len(tokens)) for tokens in token_ids]
            attention = [[1] * len(tokens) + [0] * (width - len(tokens)) for tokens in token_ids]
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention),
            }
        output: dict[str, list[list[int]] | list[int]] = {"input_ids": token_ids}
        if return_length:
            output["length"] = lengths
        return output


class FakeModel(nn.Module):
    def __init__(self, layers=4, width=3):
        super().__init__()
        self.layers = layers
        self.embedding = nn.Embedding(32, width)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, **_):
        base = input_ids.float().unsqueeze(-1).repeat(1, 1, self.embedding.embedding_dim)
        hidden_states = tuple(base + layer for layer in range(self.layers + 1))
        return SimpleNamespace(hidden_states=hidden_states)


def test_extracts_post_block_last_token_activations() -> None:
    rows = [
        {"row_id": 10, "prompt": "one two", "label": 0, "adversarial": False},
        {"row_id": 11, "prompt": "one two three four", "label": 1, "adversarial": True},
    ]

    tensors, stats = extract_activation_tensors(
        rows,
        FakeTokenizer(),
        FakeModel(),
        num_layers=4,
        hidden_size=3,
        normalized_depths=[0.25, 0.5, 1.0],
        max_length=3,
        batch_size=2,
    )

    assert stats.truncated_rows == 1
    assert stats.block_indices == (1, 2, 4)
    assert tensors["layer_25"].shape == (2, 3)
    assert torch.equal(tensors["row_ids"], torch.tensor([10, 11]))
    assert torch.all(tensors["layer_100"][0] == 6)


def test_special_token_policy_is_applied_to_length_and_batch_tokenization() -> None:
    class RecordingTokenizer(FakeTokenizer):
        def __init__(self) -> None:
            self.policies = []

        def __call__(self, prompts, *, add_special_tokens=True, **kwargs):
            self.policies.append(add_special_tokens)
            return super().__call__(
                prompts,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )

    tokenizer = RecordingTokenizer()
    extract_activation_tensors(
        [{"row_id": 1, "prompt": "one two", "label": 0}],
        tokenizer,
        FakeModel(),
        num_layers=4,
        hidden_size=3,
        normalized_depths=[1.0],
        max_length=3,
        batch_size=1,
        add_special_tokens=False,
    )

    assert tokenizer.policies == [False, False]


def test_saved_file_and_repeatability_are_verified(tmp_path: Path) -> None:
    tensors = {
        "layer_75": torch.ones((2, 3), dtype=torch.bfloat16),
        "row_ids": torch.tensor([1, 2]),
        "labels": torch.tensor([0, 1]),
    }
    path = tmp_path / "activations.safetensors"

    digest = save_activation_file(path, tensors, {"model": "fake"})
    repeated = {key: value[:1] for key, value in tensors.items()}
    assert_repeatable(tensors, repeated, rows=1, atol=0)

    assert path.is_file()
    assert len(digest) == 64
    activations, row_ids, labels = load_activation_split(path, "layer_75")
    assert activations.shape == (2, 3)
    assert row_ids.tolist() == [1, 2]
    assert labels.tolist() == [0, 1]


def test_extracts_mlp_intermediate_activations() -> None:
    model = MistralForCausalLM(
        MistralConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=32,
            attention_dropout=0.0,
        )
    ).eval()
    rows = [
        {"row_id": 1, "prompt": "one two", "label": 0},
        {"row_id": 2, "prompt": "one two three", "label": 1},
    ]

    tensors, stats = extract_activation_tensors(
        rows,
        FakeTokenizer(),
        model,
        num_layers=2,
        hidden_size=16,
        normalized_depths=[0.5],
        max_length=4,
        batch_size=2,
        activation_site="mlp_intermediate",
    )

    assert stats.block_indices == (1,)
    assert tensors["layer_50"].shape == (2, 16)
    assert torch.isfinite(tensors["layer_50"]).all()


def test_extracts_attention_output_activations() -> None:
    model = MistralForCausalLM(
        MistralConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=32,
            attention_dropout=0.0,
        )
    ).eval()
    rows = [
        {"row_id": 1, "prompt": "one two", "label": 0},
        {"row_id": 2, "prompt": "one two three", "label": 1},
    ]

    tensors, stats = extract_activation_tensors(
        rows,
        FakeTokenizer(),
        model,
        num_layers=2,
        hidden_size=8,
        normalized_depths=[0.5],
        max_length=4,
        batch_size=2,
        activation_site="attention_output",
    )

    assert stats.block_indices == (1,)
    assert tensors["layer_50"].shape == (2, 8)
    assert torch.isfinite(tensors["layer_50"]).all()


def test_bucket_uri_rejects_unsafe_paths() -> None:
    assert bucket_uri("user/bucket", "studies/run/file.safetensors") == (
        "hf://buckets/user/bucket/studies/run/file.safetensors"
    )
    with pytest.raises(ValueError, match="relative paths"):
        bucket_uri("user/bucket", "../file.safetensors")
