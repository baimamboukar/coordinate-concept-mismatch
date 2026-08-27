from typing import Any

import numpy as np
import torch

from probe_transfer.symmetry.coordinates import (
    CoordinateTransform,
    validate_permutation,
)
from probe_transfer.symmetry.coordinates import (
    inverse_permutation as _inverse_permutation,
)
from probe_transfer.symmetry.coordinates import (
    relative_permutation as _relative_permutation,
)
from probe_transfer.symmetry.scales import rescale_mlp_up_branch


def seeded_permutation(width: int, seed: int) -> torch.Tensor:
    if width < 1:
        raise ValueError("Permutation width must be positive.")
    return torch.from_numpy(np.random.default_rng(seed).permutation(width).astype(np.int64))


def seeded_gqa_head_permutation(
    query_heads: int, key_value_heads: int, head_dim: int, seed: int
) -> torch.Tensor:
    """Permute GQA groups and query heads while retaining their key/value associations."""
    _validate_attention_layout(query_heads, key_value_heads, head_dim)
    generator = np.random.default_rng(seed)
    group_size = query_heads // key_value_heads
    head_order = []
    for source_group in generator.permutation(key_value_heads):
        local_order = generator.permutation(group_size)
        head_order.extend((int(source_group) * group_size + local_order).tolist())
    return _expand_head_order(torch.tensor(head_order, dtype=torch.int64), head_dim)


def inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    return _inverse_permutation(permutation)


def relative_permutation(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _relative_permutation(current, target)


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


def permute_mlp_neurons(
    model: Any, permutation: torch.Tensor, block_indices: tuple[int, ...]
) -> None:
    """Permute SwiGLU intermediate neurons in selected blocks without changing logits."""
    validate_permutation(permutation)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if model_type not in {"llama", "mistral", "qwen3"} or layers is None:
        raise TypeError(f"Unsupported MLP-neuron permutation architecture: {model_type}")
    if not block_indices or len(block_indices) != len(set(block_indices)):
        raise ValueError("MLP-neuron permutation requires unique selected blocks.")

    with torch.no_grad():
        for block_index in block_indices:
            if block_index < 1 or block_index > len(layers):
                raise ValueError(f"Invalid MLP block index: {block_index}")
            mlp = layers[block_index - 1].mlp
            width = mlp.gate_proj.weight.shape[0]
            if len(permutation) != width or mlp.up_proj.weight.shape[0] != width:
                raise ValueError(f"Expected a width-{width} MLP-neuron permutation.")
            _permute_output(mlp.gate_proj, permutation)
            _permute_output(mlp.up_proj, permutation)
            _permute_parameter(mlp.down_proj.weight, permutation, dim=1)


def permute_attention_heads(
    model: Any, permutation: torch.Tensor, block_indices: tuple[int, ...]
) -> None:
    """Permute GQA heads before the output projection without changing block outputs."""
    validate_permutation(permutation)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if model_type not in {"llama", "mistral", "qwen3"} or layers is None:
        raise TypeError(f"Unsupported attention-head permutation architecture: {model_type}")
    if not block_indices or len(block_indices) != len(set(block_indices)):
        raise ValueError("Attention-head permutation requires unique selected blocks.")

    with torch.no_grad():
        for block_index in block_indices:
            if block_index < 1 or block_index > len(layers):
                raise ValueError(f"Invalid attention block index: {block_index}")
            attention = layers[block_index - 1].self_attn
            head_dim = int(attention.head_dim)
            query_heads = attention.q_proj.out_features // head_dim
            key_value_heads = attention.k_proj.out_features // head_dim
            _validate_attention_layout(query_heads, key_value_heads, head_dim)
            if len(permutation) != attention.q_proj.out_features:
                raise ValueError(f"Expected a width-{attention.q_proj.out_features} permutation.")
            if attention.v_proj.out_features != key_value_heads * head_dim:
                raise ValueError("Key and value projections require the same head layout.")
            if attention.o_proj.in_features != len(permutation):
                raise ValueError("Attention output projection width does not match query heads.")

            query_order = _head_order(permutation, head_dim)
            key_value_order = _key_value_order(query_order, query_heads, key_value_heads)
            key_value_permutation = _expand_head_order(key_value_order, head_dim)
            _permute_output(attention.q_proj, permutation)
            _permute_output(attention.k_proj, key_value_permutation)
            _permute_output(attention.v_proj, key_value_permutation)
            _permute_parameter(attention.o_proj.weight, permutation, dim=1)


def apply_symmetry_transform(
    model: Any,
    coordinates: CoordinateTransform,
    transformation: str,
    block_indices: tuple[int, ...],
) -> None:
    if transformation == "mlp_positive_diagonal":
        if coordinates.kind != "positive_diagonal":
            raise ValueError("MLP positive-diagonal symmetry requires positive scales.")
        rescale_mlp_up_branch(model, coordinates.values, block_indices)
        return
    if coordinates.kind != "permutation":
        raise ValueError(f"{transformation} requires a coordinate permutation.")
    permutation = coordinates.values
    if transformation == "residual_permutation":
        permute_residual(model, permutation)
    elif transformation == "mlp_neuron_permutation":
        permute_mlp_neurons(model, permutation, block_indices)
    elif transformation == "attention_head_permutation":
        permute_attention_heads(model, permutation, block_indices)
    else:
        raise ValueError(f"Unsupported symmetry transformation: {transformation}")


def _permute_output(module: Any, permutation: torch.Tensor) -> None:
    _permute_parameter(module.weight, permutation, dim=0)
    if module.bias is not None:
        _permute_parameter(module.bias, permutation, dim=0)


def _permute_parameter(parameter: torch.Tensor, permutation: torch.Tensor, *, dim: int) -> None:
    index = permutation.to(parameter.device)
    parameter.copy_(parameter.index_select(dim, index))


def _validate_attention_layout(query_heads: int, key_value_heads: int, head_dim: int) -> None:
    if min(query_heads, key_value_heads, head_dim) < 1:
        raise ValueError("Attention head counts and head dimension must be positive.")
    if query_heads % key_value_heads:
        raise ValueError("Query heads must divide evenly into key/value groups.")


def _expand_head_order(order: torch.Tensor, head_dim: int) -> torch.Tensor:
    offsets = torch.arange(head_dim, dtype=torch.int64, device=order.device)
    return (order[:, None] * head_dim + offsets).reshape(-1)


def _head_order(permutation: torch.Tensor, head_dim: int) -> torch.Tensor:
    if len(permutation) % head_dim:
        raise ValueError("Attention permutation width must be divisible by head dimension.")
    matrix = permutation.reshape(-1, head_dim)
    order = torch.div(matrix[:, 0], head_dim, rounding_mode="floor")
    if not torch.equal(matrix, _expand_head_order(order, head_dim).reshape_as(matrix)):
        raise ValueError("Attention permutations must preserve coordinates within each head.")
    validate_permutation(order)
    return order


def _key_value_order(
    query_order: torch.Tensor, query_heads: int, key_value_heads: int
) -> torch.Tensor:
    group_size = query_heads // key_value_heads
    grouped = query_order.reshape(key_value_heads, group_size)
    source_groups = torch.div(grouped, group_size, rounding_mode="floor")
    key_value_order = source_groups[:, 0]
    if not torch.equal(source_groups, key_value_order[:, None].expand_as(source_groups)):
        raise ValueError("Query-head permutation breaks grouped key/value associations.")
    local = grouped.remainder(group_size)
    expected = torch.arange(group_size, device=local.device).expand_as(local)
    if not torch.equal(torch.sort(local, dim=1).values, expected):
        raise ValueError("Each key/value group must retain all associated query heads.")
    validate_permutation(key_value_order)
    return key_value_order
