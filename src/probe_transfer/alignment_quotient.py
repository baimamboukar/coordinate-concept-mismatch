from collections.abc import Iterable

import numpy as np
import torch

from probe_transfer.alignment import AlignmentMap, fit_affine_ridge
from probe_transfer.probe_transport import StoredProbe


def build_quotient_basis(
    probes: Iterable[StoredProbe], relative_threshold: float
) -> tuple[np.ndarray, dict[str, float | int]]:
    if not 0 < relative_threshold < 1:
        raise ValueError("The quotient SVD threshold must lie between zero and one.")
    weights = np.stack([effective_linear_parameters(probe)[0] for probe in probes])
    _, singular_values, right = np.linalg.svd(weights, full_matrices=False)
    keep = singular_values > relative_threshold * singular_values[0]
    if not np.any(keep):
        raise ValueError("The quotient probe bank has no visible directions.")
    retained = singular_values[keep]
    return right[keep].astype(np.float32), {
        "quotient_rank": int(np.sum(keep)),
        "probe_bank_size": len(weights),
        "quotient_condition_number": float(retained[0] / retained[-1]),
    }


def fit_quotient_alignment(
    source: np.ndarray,
    target: np.ndarray,
    basis: np.ndarray,
    *,
    relative_alpha: float,
    device: str,
) -> AlignmentMap:
    source_values = torch.as_tensor(source, dtype=torch.float32, device=device)
    target_values = torch.as_tensor(target, dtype=torch.float32, device=device)
    quotient = torch.as_tensor(basis, dtype=torch.float32, device=device)
    return fit_affine_ridge(
        source_values @ quotient.T,
        target_values,
        relative_alpha=relative_alpha,
        method="quotient_ridge",
    )


def quotient_scores(
    probe: StoredProbe,
    quotient_values: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    weight, intercept = effective_linear_parameters(probe)
    quotient_weight = basis @ weight
    return np.asarray(quotient_values) @ quotient_weight + intercept


def effective_linear_parameters(probe: StoredProbe) -> tuple[np.ndarray, float]:
    if probe.kind != "linear":
        raise ValueError("Probe-visible quotient evaluation requires a linear probe.")
    coefficient = probe.tensors["coefficient"].squeeze(0).numpy()
    mean = probe.tensors["preprocessor.mean"].numpy()
    scale = float(probe.tensors["preprocessor.scale"].item())
    weight = coefficient / scale
    intercept = float(probe.tensors["intercept"].item() - np.dot(weight, mean))
    return weight.astype(np.float32), intercept
