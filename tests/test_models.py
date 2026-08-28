import pytest
import torch

from probe_transfer.extraction.models import (
    load_activation_model,
    openrouter_chat,
    resolve_block_indices,
    select_last_non_padding,
)


def test_huggingface_revision_must_be_pinned() -> None:
    with pytest.raises(ValueError, match="40-character commit"):
        load_activation_model("organization/model", "main")


def test_shared_tokenizer_revision_must_be_pinned() -> None:
    with pytest.raises(ValueError, match="tokenizers require"):
        load_activation_model(
            "organization/model",
            "a" * 40,
            tokenizer_id="organization/tokenizer",
            tokenizer_revision="main",
        )


def test_openrouter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        openrouter_chat("provider/model", [{"role": "user", "content": "hello"}])


def test_resolve_block_indices_uses_post_block_hidden_state_indices() -> None:
    assert resolve_block_indices(32, [0.25, 0.5, 0.75, 1.0]) == [8, 16, 24, 32]
    assert resolve_block_indices(36, [0.25, 0.5, 0.75, 1.0]) == [9, 18, 27, 36]


def test_select_last_non_padding_supports_left_and_right_padding() -> None:
    hidden = torch.arange(2 * 4 * 2).reshape(2, 4, 2)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])

    selected = select_last_non_padding(hidden, mask)

    assert torch.equal(selected, torch.stack([hidden[0, 1], hidden[1, 3]]))
