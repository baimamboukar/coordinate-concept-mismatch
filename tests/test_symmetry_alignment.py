import numpy as np
import torch

from probe_transfer.alignment.methods import (
    fit_exact_permutation_alignment,
    fit_permutation_alignment,
)
from probe_transfer.symmetry.transforms import inverse_permutation, seeded_permutation


def test_activation_estimation_recovers_unknown_permutation() -> None:
    source = np.random.default_rng(7).normal(size=(128, 16)).astype(np.float32)
    permutation = seeded_permutation(16, 42)
    target = source[:, permutation.numpy()]

    fitted = fit_permutation_alignment(source, target, device="cpu")

    assert fitted.indices is not None
    torch.testing.assert_close(fitted.indices.cpu(), inverse_permutation(permutation))
    np.testing.assert_allclose(fitted.transform(target), source, rtol=0, atol=0)


def test_exact_activation_matching_scales_without_assignment_solver() -> None:
    source = np.random.default_rng(11).normal(size=(64, 128)).astype(np.float32)
    permutation = seeded_permutation(128, 137)
    target = source[:, permutation.numpy()]

    fitted = fit_exact_permutation_alignment(source, target)

    assert fitted.method == "exact_permutation"
    assert fitted.indices is not None
    torch.testing.assert_close(fitted.indices, inverse_permutation(permutation))
    np.testing.assert_array_equal(fitted.transform(target), source)
