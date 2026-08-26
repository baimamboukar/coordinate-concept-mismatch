from typing import Any

import numpy as np
import torch


def seeded_permutation(width: int, seed: int) -> torch.Tensor:
    if width < 1:
        raise ValueError("Permutation width must be positive.")
    return torch.from_numpy(np.random.default_rng(seed).permutation(width).astype(np.int64))


def inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    validate_permutation(permutation)
    return torch.argsort(permutation)


def relative_permutation(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    validate_permutation(current)
    validate_permutation(target)
    if current.shape != target.shape:
        raise ValueError("Current and target permutations must have equal width.")
    return inverse_permutation(current).index_select(0, target)


def permute_gpt_neox_residual(model: Any, permutation: torch.Tensor) -> None:
    """Change GPT-NeoX residual coordinates while preserving its logits."""
    validate_permutation(permutation)
    neox = getattr(model, "gpt_neox", None)
    lm_head = getattr(model, "lm_head", None)
    if neox is None or lm_head is None:
        raise TypeError("Residual permutation currently supports GPT-NeoX causal LMs only.")

    width = neox.embed_in.weight.shape[1]
    if len(permutation) != width:
        raise ValueError(f"Expected a width-{width} residual permutation.")
    tied = neox.embed_in.weight.data_ptr() == lm_head.weight.data_ptr()

    with torch.no_grad():
        _permute_parameter(neox.embed_in.weight, permutation, dim=1)
        for layer in neox.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                _permute_parameter(norm.weight, permutation, dim=0)
                _permute_parameter(norm.bias, permutation, dim=0)

            _permute_parameter(layer.attention.query_key_value.weight, permutation, dim=1)
            _permute_parameter(layer.attention.dense.weight, permutation, dim=0)
            _permute_parameter(layer.attention.dense.bias, permutation, dim=0)
            _permute_parameter(layer.mlp.dense_h_to_4h.weight, permutation, dim=1)
            _permute_parameter(layer.mlp.dense_4h_to_h.weight, permutation, dim=0)
            _permute_parameter(layer.mlp.dense_4h_to_h.bias, permutation, dim=0)

        _permute_parameter(neox.final_layer_norm.weight, permutation, dim=0)
        _permute_parameter(neox.final_layer_norm.bias, permutation, dim=0)
        if not tied:
            _permute_parameter(lm_head.weight, permutation, dim=1)


def permute_mistral_residual(model: Any, permutation: torch.Tensor) -> None:
    """Change Mistral residual coordinates while preserving its logits."""
    _permute_llama_family_residual(model, permutation, expected_type="mistral")


def _permute_llama_family_residual(
    model: Any, permutation: torch.Tensor, *, expected_type: str | None = None
) -> None:
    validate_permutation(permutation)
    backbone = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if backbone is None or lm_head is None or not hasattr(backbone, "layers"):
        raise TypeError("Expected a Llama-family causal language model.")
    model_type = getattr(model.config, "model_type", None)
    if model_type not in {"llama", "mistral", "qwen3"}:
        raise TypeError(f"Unsupported Llama-family architecture: {model_type}")
    if expected_type is not None and model_type != expected_type:
        raise TypeError(f"Expected model type {expected_type}, found {model_type}.")

    embedding = backbone.embed_tokens.weight
    width = embedding.shape[1]
    if len(permutation) != width:
        raise ValueError(f"Expected a width-{width} residual permutation.")
    tied = embedding.data_ptr() == lm_head.weight.data_ptr()

    with torch.no_grad():
        _permute_parameter(embedding, permutation, dim=1)
        for layer in backbone.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                _permute_parameter(norm.weight, permutation, dim=0)

            attention = layer.self_attn
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
                _permute_parameter(projection.weight, permutation, dim=1)
            _permute_output(attention.o_proj, permutation)

            for projection in (layer.mlp.gate_proj, layer.mlp.up_proj):
                _permute_parameter(projection.weight, permutation, dim=1)
            _permute_output(layer.mlp.down_proj, permutation)

        _permute_parameter(backbone.norm.weight, permutation, dim=0)
        if not tied:
            _permute_parameter(lm_head.weight, permutation, dim=1)


def permute_residual(model: Any, permutation: torch.Tensor) -> None:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type == "gpt_neox":
        permute_gpt_neox_residual(model, permutation)
    elif model_type in {"llama", "mistral", "qwen3"}:
        _permute_llama_family_residual(model, permutation)
    else:
        raise TypeError(f"Unsupported residual-permutation architecture: {model_type}")


def _permute_output(module: Any, permutation: torch.Tensor) -> None:
    _permute_parameter(module.weight, permutation, dim=0)
    if module.bias is not None:
        _permute_parameter(module.bias, permutation, dim=0)


def _permute_parameter(parameter: torch.Tensor, permutation: torch.Tensor, *, dim: int) -> None:
    index = permutation.to(parameter.device)
    parameter.copy_(parameter.index_select(dim, index))


def validate_permutation(permutation: torch.Tensor) -> None:
    if permutation.ndim != 1 or permutation.dtype != torch.int64:
        raise ValueError("A permutation must be a one-dimensional int64 tensor.")
    expected = torch.arange(len(permutation), dtype=torch.int64, device=permutation.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("Permutation entries must contain each coordinate exactly once.")
