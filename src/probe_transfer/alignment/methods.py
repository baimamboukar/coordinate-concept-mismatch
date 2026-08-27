from dataclasses import dataclass, field
from hashlib import blake2b

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AlignmentMap:
    method: str
    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    offset: torch.Tensor | None = None
    metadata: dict[str, float | int] = field(default_factory=dict)

    def transform(self, values: np.ndarray) -> np.ndarray:
        device = _map_device(self)
        inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
        if self.indices is not None:
            outputs = inputs.index_select(1, self.indices)
            if self.scale is not None:
                outputs = outputs * self.scale
            if self.offset is not None:
                outputs = outputs + self.offset
        elif self.weight is not None and self.bias is not None:
            outputs = inputs @ self.weight + self.bias
        else:
            raise ValueError(f"Alignment map {self.method} has no transformation.")
        return outputs.detach().cpu().numpy()


def fit_ambient_alignments(
    source: np.ndarray,
    target: np.ndarray,
    *,
    relative_alpha: float,
    shuffle_seed: int,
    device: str,
) -> dict[str, AlignmentMap]:
    source_values, target_values = _paired_tensors(source, target, device)
    permutation = _fit_permutation(source_values, target_values)
    indices = permutation.indices
    if indices is None:
        raise RuntimeError("Fitted permutation is missing feature indices.")
    matched = target_values.index_select(1, indices)
    source_mean = source_values.mean(dim=0)
    target_mean = matched.mean(dim=0)
    centered_target = matched - target_mean
    covariance = (centered_target * (source_values - source_mean)).mean(dim=0)
    variance = centered_target.square().mean(dim=0).clamp_min(1e-8)
    scale = (covariance / variance).clamp_min(1e-8)

    diagonal = AlignmentMap(
        "permutation_diagonal",
        indices=indices,
        scale=scale,
        offset=source_mean - target_mean * scale,
        metadata=dict(permutation.metadata),
    )
    procrustes = _fit_procrustes(source_values, target_values)
    ridge = fit_affine_ridge(
        source_values,
        target_values,
        relative_alpha=relative_alpha,
        method="affine_ridge",
    )
    order = torch.randperm(
        len(source_values), generator=torch.Generator().manual_seed(shuffle_seed)
    ).to(source_values.device)
    shuffled = fit_affine_ridge(
        source_values.index_select(0, order),
        target_values,
        relative_alpha=relative_alpha,
        method="shuffled_affine_ridge",
    )
    return {item.method: item for item in (permutation, diagonal, procrustes, ridge, shuffled)}


def fit_permutation_alignment(
    source: np.ndarray,
    target: np.ndarray,
    *,
    device: str,
) -> AlignmentMap:
    source_values, target_values = _paired_tensors(source, target, device)
    return _fit_permutation(source_values, target_values)


def fit_exact_permutation_alignment(source: np.ndarray, target: np.ndarray) -> AlignmentMap:
    """Recover an exact feature permutation from paired, label-free activations."""
    source_values = np.asarray(source)
    target_values = np.asarray(target)
    if (
        source_values.ndim != 2
        or source_values.shape != target_values.shape
        or len(source_values) < 2
    ):
        raise ValueError("Alignment requires paired two-dimensional arrays of equal shape.")

    target_signatures: dict[bytes, list[int]] = {}
    for index in range(target_values.shape[1]):
        signature = _column_signature(target_values[:, index])
        target_signatures.setdefault(signature, []).append(index)

    indices = []
    for source_index in range(source_values.shape[1]):
        signature = _column_signature(source_values[:, source_index])
        candidates = target_signatures.get(signature, [])
        match = next(
            (
                index
                for index in candidates
                if np.array_equal(source_values[:, source_index], target_values[:, index])
            ),
            None,
        )
        if match is None:
            raise ValueError("Paired activations are not related by an exact feature permutation.")
        candidates.remove(match)
        indices.append(match)

    if any(target_signatures.values()):
        raise ValueError("Exact activation matching did not produce a bijection.")
    return AlignmentMap(
        "exact_permutation",
        indices=torch.tensor(indices, dtype=torch.int64),
        metadata={
            "matched_correlation_mean": 1.0,
            "matched_correlation_minimum": 1.0,
        },
    )


def fit_affine_ridge(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    relative_alpha: float,
    method: str,
) -> AlignmentMap:
    if relative_alpha <= 0:
        raise ValueError("Relative Ridge alpha must be positive.")
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    gram = target_centered.T @ target_centered
    scale = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
    penalty = relative_alpha * scale
    regularized = gram + penalty * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    weight = torch.linalg.solve(regularized, target_centered.T @ source_centered)
    return AlignmentMap(
        method,
        weight=weight,
        bias=source_mean - target_mean @ weight,
        metadata={"ridge_penalty": float(penalty.item())},
    )


def alignment_diagnostic(
    alignment: AlignmentMap,
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    predicted = alignment.transform(target).astype(np.float64)
    expected = np.asarray(source, dtype=np.float64)
    residual_rmse = float(np.sqrt(np.mean((predicted - expected) ** 2)))
    denominator = float(np.sqrt(np.mean((expected - expected.mean(axis=0)) ** 2)))
    dot = np.sum(predicted * expected, axis=1)
    norms = np.linalg.norm(predicted, axis=1) * np.linalg.norm(expected, axis=1)
    return {
        "alignment_relative_rmse": residual_rmse / max(denominator, 1e-12),
        "alignment_mean_cosine": float(np.mean(dot / np.maximum(norms, 1e-12))),
    }


def _fit_procrustes(source: torch.Tensor, target: torch.Tensor) -> AlignmentMap:
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    cross_covariance = (target - target_mean).T @ (source - source_mean)
    left, _, right = torch.linalg.svd(cross_covariance, full_matrices=False)
    weight = left @ right
    return AlignmentMap(
        "orthogonal_procrustes",
        weight=weight,
        bias=source_mean - target_mean @ weight,
    )


def _feature_matches(
    source: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    source_z = _standardize(source)
    target_z = _standardize(target)
    correlations = target_z.T @ source_z / len(source_z)
    target_indices, source_indices = linear_sum_assignment(
        correlations.detach().cpu().numpy(), maximize=True
    )
    order = np.argsort(source_indices)
    indices = torch.as_tensor(target_indices[order], device=target.device)
    matched = correlations[indices, torch.arange(source.shape[1], device=source.device)]
    return indices, matched


def _fit_permutation(source: torch.Tensor, target: torch.Tensor) -> AlignmentMap:
    indices, correlations = _feature_matches(source, target)
    return AlignmentMap(
        "permutation",
        indices=indices,
        metadata=_correlation_metadata(correlations),
    )


def _standardize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0)
    return centered / centered.square().mean(dim=0).sqrt().clamp_min(1e-8)


def _paired_tensors(
    source: np.ndarray, target: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.ndim != 2 or source.shape != target.shape or len(source) < 2:
        raise ValueError("Alignment requires paired two-dimensional arrays of equal shape.")
    return (
        torch.as_tensor(source, dtype=torch.float32, device=device),
        torch.as_tensor(target, dtype=torch.float32, device=device),
    )


def _correlation_metadata(values: torch.Tensor) -> dict[str, float]:
    return {
        "matched_correlation_mean": float(values.mean().item()),
        "matched_correlation_minimum": float(values.min().item()),
    }


def _column_signature(values: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(values)
    return blake2b(contiguous.tobytes(), digest_size=16).digest()


def _map_device(alignment: AlignmentMap) -> torch.device:
    for value in (alignment.weight, alignment.indices, alignment.scale):
        if value is not None:
            return value.device
    return torch.device("cpu")
