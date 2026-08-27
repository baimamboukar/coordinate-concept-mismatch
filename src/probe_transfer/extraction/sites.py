from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import torch

from probe_transfer.extraction.models import select_last_non_padding

RESIDUAL_STREAM = "residual_stream"
MLP_INTERMEDIATE = "mlp_intermediate"
ATTENTION_OUTPUT = "attention_output"
ACTIVATION_SITES = frozenset({RESIDUAL_STREAM, MLP_INTERMEDIATE, ATTENTION_OUTPUT})


def activation_width(activations: dict[str, Any], model: dict[str, Any]) -> int:
    site = activations.get("site", RESIDUAL_STREAM)
    key = "intermediate_size" if site == MLP_INTERMEDIATE else "hidden_size"
    width = model.get(key)
    if site not in ACTIVATION_SITES or not isinstance(width, int) or width < 1:
        raise ValueError(f"Model requires a positive {key} for activation site {site}.")
    return width


def activation_boundary(site: str) -> str:
    if site == RESIDUAL_STREAM:
        return "transformers_output_hidden_states"
    if site == MLP_INTERMEDIATE:
        return "mlp_down_projection_input"
    if site == ATTENTION_OUTPUT:
        return "attention_output_projection_input"
    raise ValueError(f"Unsupported activation site: {site}")


class ActivationCapture:
    def __init__(self, model: Any, block_indices: Sequence[int], site: str) -> None:
        if site not in ACTIVATION_SITES:
            raise ValueError(f"Unsupported activation site: {site}")
        self.model = model
        self.block_indices = tuple(int(index) for index in block_indices)
        self.site = site
        self._captured: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    @property
    def requires_hidden_states(self) -> bool:
        return self.site == RESIDUAL_STREAM

    def __enter__(self) -> Self:
        if self.site != RESIDUAL_STREAM:
            layers = _decoder_layers(self.model)
            for block_index in self.block_indices:
                if block_index < 1 or block_index > len(layers):
                    raise ValueError(f"Invalid decoder block index: {block_index}")
                layer = layers[block_index - 1]
                projection = (
                    layer.mlp.down_proj if self.site == MLP_INTERMEDIATE else layer.self_attn.o_proj
                )
                self._handles.append(projection.register_forward_pre_hook(self._hook(block_index)))
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()

    def clear(self) -> None:
        self._captured.clear()

    def selected(self, output: Any, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        if self.site == RESIDUAL_STREAM:
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise ValueError("Residual activation capture requires hidden states.")
            states = [hidden_states[index] for index in self.block_indices]
        else:
            missing = set(self.block_indices) - set(self._captured)
            if missing:
                raise RuntimeError(f"{self.site} hooks did not run for blocks: {sorted(missing)}")
            states = [self._captured[index] for index in self.block_indices]
        return [select_last_non_padding(state, attention_mask) for state in states]

    def _hook(self, block_index: int):
        def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("Activation boundary must receive one tensor.")
            if block_index in self._captured:
                raise RuntimeError(f"Decoder block {block_index} ran more than once.")
            self._captured[block_index] = inputs[0]

        return capture


def _decoder_layers(model: Any) -> Any:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("Hooked activation capture requires a Llama-family decoder.")
    return layers
