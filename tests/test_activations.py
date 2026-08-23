from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from probe_transfer.activations import (
    activation_output_directory,
    assert_repeatable,
    bucket_uri,
    extract_activation_tensors,
    save_activation_file,
    upload_bucket_file,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(
        self,
        prompts,
        *,
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


def test_deferred_upload_uses_configured_staging_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACTIVATION_STAGING_DIR", str(tmp_path))

    with activation_output_directory(True) as staging_dir:
        assert staging_dir == tmp_path.resolve()


def test_bucket_uri_rejects_unsafe_paths() -> None:
    assert bucket_uri("user/bucket", "experiments/run/file.safetensors") == (
        "hf://buckets/user/bucket/experiments/run/file.safetensors"
    )
    with pytest.raises(ValueError, match="relative paths"):
        bucket_uri("user/bucket", "../file.safetensors")


def test_upload_uses_and_verifies_the_expected_bucket_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "llama.safetensors"
    path.write_bytes(b"activation data")
    commands = []

    def fake_run(command, **_):
        commands.append(command)
        listing = "seed_42/llama.safetensors\n" if command[2] == "list" else ""
        return SimpleNamespace(stdout=listing)

    monkeypatch.setattr("probe_transfer.activations.shutil.which", lambda _: "/usr/bin/hf")
    monkeypatch.setattr("probe_transfer.activations.subprocess.run", fake_run)

    artifact = upload_bucket_file(
        path,
        "user/bucket",
        "experiments/baseline/activations/smoke/seed_42/llama.safetensors",
    )

    assert artifact.uri.endswith("seed_42/llama.safetensors")
    assert commands[0][1:3] == ["buckets", "cp"]
    assert commands[1][1:3] == ["buckets", "list"]
