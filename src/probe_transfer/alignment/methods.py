from dataclasses import dataclass, field
from hashlib import blake2b

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from probe_transfer.alignment.ridge import RidgeSystem


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
    methods: list[str] | None = None,
) -> dict[str, AlignmentMap]:
    source_values, target_values = _paired_tensors(source, target, device)
    available = {
        "permutation",
        "permutation_diagonal",
        "orthogonal_procrustes",
        "affine_ridge",
        "shuffled_affine_ridge",
    }
    selected = available if methods is None else set(methods)
    if selected - available:
        raise ValueError("Unsupported ambient alignment method requested.")
    fitted = {}
    if selected & {"permutation", "permutation_diagonal"}:
        permutation = _fit_permutation(source_values, target_values)
        if "permutation" in selected:
            fitted[permutation.method] = permutation
        if "permutation_diagonal" in selected:
            fitted["permutation_diagonal"] = _fit_permutation_diagonal(
                source_values, target_values, permutation
            )
    if "orthogonal_procrustes" in selected:
        fitted["orthogonal_procrustes"] = _fit_procrustes(source_values, target_values)
    if "affine_ridge" in selected:
        fitted["affine_ridge"] = fit_affine_ridge(
            source_values,
            target_values,
            relative_alpha=relative_alpha,
            method="affine_ridge",
        )
    if "shuffled_affine_ridge" in selected:
        order = torch.randperm(
            len(source_values), generator=torch.Generator().manual_seed(shuffle_seed)
        ).to(source_values.device)
        fitted["shuffled_affine_ridge"] = fit_affine_ridge(
            source_values.index_select(0, order),
            target_values,
            relative_alpha=relative_alpha,
            method="shuffled_affine_ridge",
        )
    return fitted


def _fit_permutation_diagonal(
    source: torch.Tensor, target: torch.Tensor, permutation: AlignmentMap
) -> AlignmentMap:
    indices = permutation.indices
    if indices is None:
        raise RuntimeError("Fitted permutation is missing feature indices.")
    matched = target.index_select(1, indices)
    source_mean = source.mean(dim=0)
    target_mean = matched.mean(dim=0)
    centered_target = matched - target_mean
    covariance = (centered_target * (source - source_mean)).mean(dim=0)
    variance = centered_target.square().mean(dim=0).clamp_min(1e-8)
    scale = (covariance / variance).clamp_min(1e-8)
    return AlignmentMap(
        "permutation_diagonal",
        indices=indices,
        scale=scale,
        offset=source_mean - target_mean * scale,
        metadata=dict(permutation.metadata),
    )


def fit_permutation_alignment(
    source: np.ndarray, target: np.ndarray, *, device: str
) -> AlignmentMap:
    source_values, target_values = _paired_tensors(source, target, device)
    return _fit_permutation(source_values, target_values)


def fit_exact_permutation_alignment(source: np.ndarray, target: np.ndarray) -> AlignmentMap:
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


def fit_positive_diagonal_alignment(
    source: np.ndarray, target: np.ndarray, *, relative_tolerance: float
) -> AlignmentMap:
    if relative_tolerance <= 0:
        raise ValueError("Positive-diagonal fit tolerance must be positive.")
    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if (
        source_values.ndim != 2
        or source_values.shape != target_values.shape
        or len(source_values) < 2
    ):
        raise ValueError("Alignment requires paired two-dimensional arrays of equal shape.")
    denominator = np.sum(target_values * target_values, axis=0)
    if np.any(denominator <= np.finfo(np.float64).tiny):
        raise ValueError("Positive-diagonal alignment contains an unidentifiable feature.")
    inverse_scale = np.sum(target_values * source_values, axis=0) / denominator
    if np.any(~np.isfinite(inverse_scale)) or np.any(inverse_scale <= 0):
        raise ValueError("Estimated diagonal map is not finite and strictly positive.")
    residual = target_values * inverse_scale - source_values
    relative_error = np.linalg.norm(residual) / max(np.linalg.norm(source_values), 1e-12)
    if relative_error > relative_tolerance:
        raise ValueError("Paired activations are not related by a positive diagonal map.")
    return AlignmentMap(
        "positive_diagonal",
        indices=torch.arange(source_values.shape[1], dtype=torch.int64),
        scale=torch.from_numpy(inverse_scale.astype(np.float32)),
        metadata={"fit_relative_error": float(relative_error)},
    )


def fit_affine_ridge(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    relative_alpha: float,
    method: str,
) -> AlignmentMap:
    weight, bias, penalty = RidgeSystem.prepare(source, target).solve(relative_alpha)
    return AlignmentMap(
        method,
        weight=weight,
        bias=bias,
        metadata={"ridge_penalty": penalty},
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
