from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch

from core.config import ConfigError
from probe_transfer.alignment.methods import AlignmentMap


@dataclass(frozen=True)
class RidgeFactor:
    inputs: torch.Tensor
    penalty: float
    factor: torch.Tensor
    input_q: torch.Tensor | None
    input_r: torch.Tensor | None

    @classmethod
    def prepare(cls, inputs: torch.Tensor, relative_alpha: float) -> "RidgeFactor":
        scale = inputs.square().sum(dim=0).mean().clamp_min(torch.finfo(inputs.dtype).eps)
        penalty = relative_alpha * scale
        if len(inputs) < inputs.shape[1]:
            regularized = inputs @ inputs.T + penalty * torch.eye(
                len(inputs), dtype=inputs.dtype, device=inputs.device
            )
            input_q, input_r = torch.linalg.qr(inputs.T, mode="reduced")
        else:
            regularized = inputs.T @ inputs + penalty * torch.eye(
                inputs.shape[1], dtype=inputs.dtype, device=inputs.device
            )
            input_q = input_r = None
        return cls(
            inputs,
            float(penalty.item()),
            torch.linalg.cholesky(regularized),
            input_q,
            input_r,
        )

    def updates(self, outputs: torch.Tensor, ranks: list[int]) -> dict[int, torch.Tensor]:
        if self.input_q is not None and self.input_r is not None:
            coefficients = torch.cholesky_solve(outputs, self.factor)
            output_q, output_r = torch.linalg.qr(coefficients.T, mode="reduced")
            left, singular, right = torch.linalg.svd(self.input_r @ output_r.T, full_matrices=False)
            left = self.input_q @ left
            right = right @ output_q.T
        else:
            correction = torch.cholesky_solve(self.inputs.T @ outputs, self.factor)
            left, singular, right = torch.linalg.svd(correction, full_matrices=False)
        if not torch.isfinite(singular).all():
            raise ValueError("Low-rank ridge correction produced a non-finite map.")
        return {rank: (left[:, :rank] * singular[:rank]) @ right[:rank] for rank in ranks}


def fit_coral(
    base: AlignmentMap,
    source: torch.Tensor,
    target: torch.Tensor,
    rank: int,
    rows: int,
    shrinkage: float,
    method: str,
) -> AlignmentMap:
    if base.weight is None or base.bias is None:
        raise ValueError("CORAL requires a complete affine base map.")
    mapped = target @ base.weight + base.bias
    transport = _matrix_power(_covariance(mapped, shrinkage), -0.5) @ _matrix_power(
        _covariance(source, shrinkage), 0.5
    )
    delta = base.weight @ transport - base.weight
    left, singular, right = torch.linalg.svd(delta, full_matrices=False)
    update = (left[:, :rank] * singular[:rank]) @ right[:rank]
    weight = base.weight + update
    bias = source.mean(0) - target.mean(0) @ weight
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("CORAL adaptation produced a non-finite map.")
    return AlignmentMap(
        method,
        weight=weight,
        bias=bias,
        metadata={
            "calibration_rows": rows,
            "correction_rank": rank,
            "covariance_shrinkage": shrinkage,
            "row_correspondence": 0,
            "correction_frobenius_norm": float(torch.linalg.vector_norm(update).item()),
            "bias_shift_norm": float(torch.linalg.vector_norm(bias - base.bias).item()),
        },
    )


def shuffle_order(rows: int, seed: int, device: torch.device) -> torch.Tensor:
    order = torch.randperm(rows, generator=torch.Generator().manual_seed(seed))
    if rows > 1 and torch.equal(order, torch.arange(rows)):
        order = order.roll(1)
    return order.to(device)


def repeat_seed(seed: int, rows: int, repeat: int) -> int:
    return seed + rows + 104_729 * (repeat + 1)


def validate_control_settings(controls: Any, rows: list[int]) -> None:
    required = {
        "residual_shuffle_repeats",
        "source_shuffle_repeats",
        "moment_matching",
        "covariance_shrinkage",
    }
    if not isinstance(controls, dict) or set(controls) != required:
        raise ConfigError("Task-adaptation controls require the complete decomposition contract.")
    repeats = (controls["residual_shuffle_repeats"], controls["source_shuffle_repeats"])
    shrinkage = controls["covariance_shrinkage"]
    if any(type(value) is not int or value < 1 for value in repeats):
        raise ConfigError("Control repeat counts must be positive integers.")
    if controls["moment_matching"] != "coral":
        raise ConfigError("The supported unpaired moment-matching control is CORAL.")
    if (
        type(shrinkage) not in {int, float}
        or not isfinite(shrinkage)
        or not 0 < shrinkage <= 1
        or rows[0] < 2
    ):
        raise ConfigError("CORAL requires at least two rows and shrinkage in (0, 1].")


def _covariance(values: torch.Tensor, shrinkage: float) -> torch.Tensor:
    centered = values - values.mean(0)
    covariance = centered.T @ centered / (len(values) - 1)
    scale = torch.trace(covariance) / covariance.shape[0]
    identity = torch.eye(covariance.shape[0], dtype=values.dtype, device=values.device)
    return (1 - shrinkage) * covariance + shrinkage * scale * identity


def _matrix_power(matrix: torch.Tensor, exponent: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix)
    floor = torch.finfo(matrix.dtype).eps * values.abs().max().clamp_min(1)
    return (vectors * values.clamp_min(floor).pow(exponent)) @ vectors.T
