import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from probe_transfer.extraction.models import resolve_block_indices
from probe_transfer.extraction.sites import RESIDUAL_STREAM, ActivationCapture


@dataclass(frozen=True)
class ActivationStats:
    rows: int
    truncated_rows: int
    block_indices: tuple[int, ...]

    @property
    def truncation_rate(self) -> float:
        return self.truncated_rows / self.rows


def extract_activation_tensors(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    *,
    num_layers: int,
    hidden_size: int,
    normalized_depths: list[float],
    max_length: int,
    batch_size: int,
    add_special_tokens: bool = True,
    activation_site: str = RESIDUAL_STREAM,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, torch.Tensor], ActivationStats]:
    if not rows:
        raise ValueError("Activation extraction requires at least one row.")
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a padding or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    block_indices = resolve_block_indices(num_layers, normalized_depths)
    layer_keys = [_layer_key(depth) for depth in normalized_depths]
    tensors = {
        key: torch.empty((len(rows), hidden_size), dtype=storage_dtype) for key in layer_keys
    }
    tensors["row_ids"] = torch.tensor([int(row["row_id"]) for row in rows])
    tensors["labels"] = torch.tensor([int(row["label"]) for row in rows])
    tensors["adversarial"] = torch.tensor(
        [_encode_adversarial(row.get("adversarial")) for row in rows], dtype=torch.int8
    )

    input_device = model.get_input_embeddings().weight.device
    truncated_rows = 0
    with ActivationCapture(model, block_indices, activation_site) as capture:
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            prompts = [row["prompt"] for row in rows[start:stop]]
            lengths = _token_lengths(tokenizer, prompts, add_special_tokens)
            truncated_rows += sum(length > max_length for length in lengths)
            encoded = tokenizer(
                prompts,
                add_special_tokens=add_special_tokens,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            model_inputs = {name: value.to(input_device) for name, value in encoded.items()}
            capture.clear()
            with torch.inference_mode():
                outputs = model(
                    **model_inputs,
                    output_hidden_states=capture.requires_hidden_states,
                    use_cache=False,
                    return_dict=True,
                )
            selected_states = capture.selected(outputs, model_inputs["attention_mask"])
            for key, block_index, selected in zip(
                layer_keys, block_indices, selected_states, strict=True
            ):
                if selected.shape != (stop - start, hidden_size):
                    raise ValueError(
                        f"Block {block_index} produced {tuple(selected.shape)}, expected "
                        f"{(stop - start, hidden_size)}."
                    )
                tensors[key][start:stop].copy_(selected.to(device="cpu", dtype=storage_dtype))

    finite = all(
        torch.isfinite(tensor).all() for key, tensor in tensors.items() if key.startswith("layer_")
    )
    if not finite:
        raise ValueError("Extracted activations contain non-finite values.")
    return tensors, ActivationStats(len(rows), truncated_rows, tuple(block_indices))


def assert_repeatable(
    reference: dict[str, torch.Tensor],
    repeated: dict[str, torch.Tensor],
    *,
    rows: int,
    atol: float,
) -> None:
    for key in reference:
        expected = reference[key][:rows]
        actual = repeated[key][:rows]
        if key.startswith("layer_"):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)
        elif not torch.equal(actual, expected):
            raise AssertionError(f"Repeated extraction changed {key}.")


def save_activation_file(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = {key: tensor.contiguous() for key, tensor in tensors.items()}
    save_file(contiguous, path, metadata=metadata)
    with safe_open(path, framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(contiguous):
            raise RuntimeError("Saved activation keys do not match the extracted tensors.")
        for key, tensor in contiguous.items():
            if tuple(saved.get_slice(key).get_shape()) != tuple(tensor.shape):
                raise RuntimeError(f"Saved shape mismatch for {key}.")
    return _sha256(path)


def load_activation_split(
    path: str | Path, layer_key: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = {layer_key, "row_ids", "labels"}
    with safe_open(path, framework="pt", device="cpu") as saved:
        missing = required - set(saved.keys())
        if missing:
            raise ValueError(f"Activation file is missing tensors: {sorted(missing)}")
        activations = saved.get_tensor(layer_key).float()
        row_ids = saved.get_tensor("row_ids").long()
        labels = saved.get_tensor("labels").long()
    if len(activations) != len(row_ids) or len(row_ids) != len(labels):
        raise ValueError("Activation, row ID, and label counts do not match.")
    return activations, row_ids, labels


def prompt_truncation_rate(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    add_special_tokens: bool,
    batch_size: int = 256,
) -> float:
    if not rows or max_length < 1 or batch_size < 1:
        raise ValueError("Prompt audit requires rows and positive length and batch settings.")
    truncated = 0
    for start in range(0, len(rows), batch_size):
        prompts = [row["prompt"] for row in rows[start : start + batch_size]]
        truncated += sum(
            length > max_length for length in _token_lengths(tokenizer, prompts, add_special_tokens)
        )
    return truncated / len(rows)


def _token_lengths(tokenizer: Any, prompts: list[str], add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(
        prompts,
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
        return_length=True,
    )
    lengths = encoded.get("length")
    if lengths is not None:
        return [int(length) for length in lengths]
    return [len(token_ids) for token_ids in encoded["input_ids"]]


def _layer_key(depth: float) -> str:
    return f"layer_{round(depth * 100)}"


def _encode_adversarial(value: Any) -> int:
    return -1 if value is None else int(bool(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
