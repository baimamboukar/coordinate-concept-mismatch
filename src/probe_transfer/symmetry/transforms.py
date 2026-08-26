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


def _permute_parameter(parameter: torch.Tensor, permutation: torch.Tensor, *, dim: int) -> None:
    index = permutation.to(parameter.device)
    parameter.copy_(parameter.index_select(dim, index))


def validate_permutation(permutation: torch.Tensor) -> None:
    if permutation.ndim != 1 or permutation.dtype != torch.int64:
        raise ValueError("A permutation must be a one-dimensional int64 tensor.")
    expected = torch.arange(len(permutation), dtype=torch.int64, device=permutation.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("Permutation entries must contain each coordinate exactly once.")
