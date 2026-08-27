from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

CoordinateKind = Literal["permutation", "positive_diagonal"]


@dataclass(frozen=True)
class CoordinateTransform:
    kind: CoordinateKind
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.kind == "permutation":
            validate_permutation(self.values)
        elif self.kind == "positive_diagonal":
            validate_positive_diagonal(self.values)
        else:
            raise ValueError(f"Unsupported coordinate transform: {self.kind}")

    @classmethod
    def identity(cls, kind: CoordinateKind, width: int) -> "CoordinateTransform":
        if width < 1:
            raise ValueError("Coordinate width must be positive.")
        values = (
            torch.arange(width, dtype=torch.int64)
            if kind == "permutation"
            else torch.ones(width, dtype=torch.float64)
        )
        return cls(kind, values)

    def apply_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        self._require_width(inputs.shape[-1])
        if self.kind == "permutation":
            return inputs.index_select(-1, self.values.to(inputs.device))
        scales = self.values.to(device=inputs.device, dtype=inputs.dtype)
        return inputs * scales

    def apply_array(self, inputs: np.ndarray) -> np.ndarray:
        self._require_width(inputs.shape[-1])
        values = self.values.cpu().numpy()
        if self.kind == "permutation":
            return inputs[..., values]
        return inputs * values.astype(inputs.dtype, copy=False)

    def inverse(self) -> "CoordinateTransform":
        values = (
            inverse_permutation(self.values)
            if self.kind == "permutation"
            else self.values.reciprocal()
        )
        return CoordinateTransform(self.kind, values)

    @property
    def artifact_label(self) -> str:
        return "permutation" if self.kind == "permutation" else "scale"

    @property
    def map_filename(self) -> str:
        return "permutations.json" if self.kind == "permutation" else "scales.json"

    @property
    def raw_condition(self) -> str:
        return "raw_permuted" if self.kind == "permutation" else "raw_rescaled"

    def relative_from(self, current: "CoordinateTransform") -> "CoordinateTransform":
        if self.kind != current.kind or len(self.values) != len(current.values):
            raise ValueError("Coordinate transforms must have equal kind and width.")
        values = (
            relative_permutation(current.values, self.values)
            if self.kind == "permutation"
            else self.values / current.values
        )
        return CoordinateTransform(self.kind, values)

    def _require_width(self, width: int) -> None:
        if len(self.values) != width:
            raise ValueError(f"Expected coordinate width {len(self.values)}, found {width}.")


def validate_permutation(permutation: torch.Tensor) -> None:
    if permutation.ndim != 1 or permutation.dtype != torch.int64:
        raise ValueError("A permutation must be a one-dimensional int64 tensor.")
    expected = torch.arange(len(permutation), dtype=torch.int64, device=permutation.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("Permutation entries must contain each coordinate exactly once.")


def validate_positive_diagonal(scales: torch.Tensor) -> None:
    if scales.ndim != 1 or not scales.is_floating_point():
        raise ValueError("Positive-diagonal scales must be a one-dimensional floating tensor.")
    if not bool(torch.all(torch.isfinite(scales) & (scales > 0))):
        raise ValueError("Positive-diagonal scales must be finite and strictly positive.")


def inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    validate_permutation(permutation)
    return torch.argsort(permutation)


def relative_permutation(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    validate_permutation(current)
    validate_permutation(target)
    if current.shape != target.shape:
        raise ValueError("Current and target permutations must have equal width.")
    return inverse_permutation(current).index_select(0, target)
