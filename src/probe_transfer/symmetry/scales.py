from typing import Any

import numpy as np
import torch

from probe_transfer.symmetry.coordinates import validate_positive_diagonal


def seeded_positive_diagonal(width: int, seed: int, minimum: float, maximum: float) -> torch.Tensor:
    if width < 1 or minimum <= 0 or maximum <= minimum:
        raise ValueError("Positive-diagonal sampling requires a valid width and scale range.")
    generator = np.random.default_rng(seed)
    values = np.exp(generator.uniform(np.log(minimum), np.log(maximum), width))
    return torch.from_numpy(values)


def rescale_mlp_up_branch(model: Any, scales: torch.Tensor, block_indices: tuple[int, ...]) -> None:
    validate_positive_diagonal(scales)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if model_type not in {"llama", "mistral", "qwen3"} or layers is None:
        raise TypeError(f"Unsupported MLP positive-diagonal architecture: {model_type}")
    if not block_indices or len(block_indices) != len(set(block_indices)):
        raise ValueError("MLP positive-diagonal symmetry requires unique selected blocks.")

    with torch.no_grad():
        for block_index in block_indices:
            if block_index < 1 or block_index > len(layers):
                raise ValueError(f"Invalid MLP block index: {block_index}")
            mlp = layers[block_index - 1].mlp
            width = mlp.up_proj.out_features
            if (
                len(scales) != width
                or mlp.gate_proj.out_features != width
                or mlp.down_proj.in_features != width
            ):
                raise ValueError(f"Expected {width} positive-diagonal scales.")
            values = scales.to(device=mlp.up_proj.weight.device, dtype=mlp.up_proj.weight.dtype)
            mlp.up_proj.weight.mul_(values[:, None])
            if mlp.up_proj.bias is not None:
                mlp.up_proj.bias.mul_(values)
            mlp.down_proj.weight.div_(values[None, :])
