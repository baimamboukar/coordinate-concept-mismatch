import gc
from dataclasses import dataclass
from typing import Any

import torch

from probe_transfer.extraction.models import (
    load_activation_model,
    resolve_block_indices,
    select_last_non_padding,
)
from probe_transfer.extraction.runtime import validate_loaded_model
from probe_transfer.extraction.sites import ActivationCapture, activation_width
from probe_transfer.symmetry.protocol import selected_models
from probe_transfer.symmetry.transforms import apply_symmetry_permutation, relative_permutation


@dataclass(frozen=True)
class GateOutputs:
    logits: torch.Tensor
    hidden_states: dict[str, torch.Tensor]


def run_function_gates(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    permutations: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    symmetry = config["symmetry"]
    records = []
    for model_name in selected_models(config):
        model_config = config["models"][model_name]
        tokenizer, model = load_activation_model(
            model_config["id"],
            model_config["revision"],
            dtype=symmetry["gate_dtype"],
        )
        validate_loaded_model(
            model,
            layers=model_config["layers"],
            hidden_size=model_config["hidden_size"],
        )
        site = config["activations"].get("site", "residual_stream")
        reference = _collect_outputs(tokenizer, model, rows, model_config, symmetry, site)
        width = activation_width(config["activations"], model_config)
        blocks = tuple(resolve_block_indices(model_config["layers"], symmetry["probed_depths"]))
        current = torch.arange(width)
        targets: list[tuple[str, int | None, torch.Tensor]] = [
            ("identity", None, torch.arange(width)),
            *[("permutation", seed, permutation) for seed, permutation in permutations.items()],
        ]
        for condition, seed, target in targets:
            apply_symmetry_permutation(
                model,
                relative_permutation(current, target),
                symmetry["transformation"],
                blocks,
            )
            actual = _collect_outputs(tokenizer, model, rows, model_config, symmetry, site)
            records.append(
                _gate_record(
                    model_name,
                    condition,
                    seed,
                    reference,
                    actual,
                    target,
                    symmetry,
                    activation_site=site,
                )
            )
            current = target

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def _collect_outputs(
    tokenizer: Any,
    model: Any,
    rows: list[dict[str, Any]],
    model_config: dict[str, Any],
    symmetry: dict[str, Any],
    activation_site: str = "residual_stream",
) -> GateOutputs:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks = resolve_block_indices(model_config["layers"], symmetry["probed_depths"])
    keys = [f"layer_{round(depth * 100)}" for depth in symmetry["probed_depths"]]
    logits = []
    hidden_states = {key: [] for key in keys}
    input_device = model.get_input_embeddings().weight.device

    with ActivationCapture(model, blocks, activation_site) as capture:
        for start in range(0, len(rows), symmetry["gate_batch_size"]):
            prompts = [row["prompt"] for row in rows[start : start + symmetry["gate_batch_size"]]]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=symmetry["gate_max_length"],
                return_tensors="pt",
            )
            inputs = {name: value.to(input_device) for name, value in encoded.items()}
            capture.clear()
            with torch.inference_mode():
                output = model(
                    **inputs,
                    output_hidden_states=capture.requires_hidden_states,
                    use_cache=False,
                    return_dict=True,
                )
            selected = capture.selected(output, inputs["attention_mask"])
            logits.append(select_last_non_padding(output.logits, inputs["attention_mask"]).cpu())
            for key, values in zip(keys, selected, strict=True):
                hidden_states[key].append(values.cpu())
    return GateOutputs(
        logits=torch.cat(logits),
        hidden_states={key: torch.cat(values) for key, values in hidden_states.items()},
    )


def _gate_record(
    model: str,
    condition: str,
    seed: int | None,
    reference: GateOutputs,
    actual: GateOutputs,
    target: torch.Tensor,
    config: dict[str, Any],
    *,
    activation_site: str = "residual_stream",
) -> dict[str, Any]:
    absolute_errors = torch.abs(reference.logits - actual.logits)
    tolerances = config["logit_atol"] + config["logit_rtol"] * torch.abs(reference.logits)
    logit_close = bool(torch.all(absolute_errors <= tolerances))
    logit_error = float(torch.max(absolute_errors))
    scaled_logit_error = float(torch.max(absolute_errors / tolerances))
    agreement = float(
        torch.mean((reference.logits.argmax(dim=-1) == actual.logits.argmax(dim=-1)).float())
    )
    activation_errors = {}
    for key, expected in reference.hidden_states.items():
        permuted = expected.index_select(1, target)
        error = torch.linalg.vector_norm(actual.hidden_states[key] - permuted)
        denominator = torch.clamp(torch.linalg.vector_norm(permuted), min=1e-12)
        activation_errors[key] = float(error / denominator)
    maximum_activation_error = max(activation_errors.values())
    passed = bool(
        logit_close
        and agreement == 1.0
        and maximum_activation_error <= config["activation_relative_tolerance"]
    )
    return {
        "model": model,
        "condition": condition,
        "permutation_seed": seed,
        "rows": len(reference.logits),
        "logit_position": "last_non_padding",
        "activation_site": activation_site,
        "dtype": config["gate_dtype"],
        "maximum_logit_error": logit_error,
        "maximum_scaled_logit_error": scaled_logit_error,
        "logit_tolerance_passed": logit_close,
        "next_token_agreement": agreement,
        "activation_relative_errors": activation_errors,
        "maximum_activation_relative_error": maximum_activation_error,
        "passed": passed,
    }
