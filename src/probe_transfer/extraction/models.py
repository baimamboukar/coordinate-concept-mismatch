import os
from collections.abc import Sequence
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from core.constants import OPENROUTER_BASE_URL
from core.reproducibility import is_pinned_hf_revision


def resolve_block_indices(num_layers: int, normalized_depths: Sequence[float]) -> list[int]:
    indices = []
    for depth in normalized_depths:
        if not 0 < depth <= 1:
            raise ValueError("Normalized depths must lie in (0, 1].")
        index = round(num_layers * depth)
        if index < 1 or index > num_layers:
            raise ValueError(f"Depth {depth} does not resolve to a transformer block.")
        indices.append(index)
    return indices


def select_last_non_padding(hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    if hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected hidden states [batch, tokens, width] and a 2D mask.")
    if hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden-state and attention-mask dimensions do not match.")

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1)
    last_positions = positions.max(dim=1).values
    if torch.any(last_positions < 0):
        raise ValueError("Every example must contain at least one non-padding token.")
    batch = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch, last_positions.to(hidden_state.device)]


def load_activation_model(
    model_id: str,
    revision: str,
    *,
    dtype: str = "bfloat16",
    device_map: str | dict[str, Any] = "auto",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    if not is_pinned_hf_revision(revision):
        raise ValueError("Hugging Face models require an exact 40-character commit revision.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    common = {
        "revision": revision,
        "token": os.getenv("HF_TOKEN"),
        "trust_remote_code": trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device_map,
        **common,
    )
    model.config.output_hidden_states = True
    model.eval()
    return tokenizer, model


def openrouter_chat(
    model: str,
    messages: Sequence[dict[str, str]],
    **parameters: Any,
) -> Any:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for black-box inference.")

    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return client.chat.completions.create(
        model=model,
        messages=cast(list[ChatCompletionMessageParam], list(messages)),
        **parameters,
    )
