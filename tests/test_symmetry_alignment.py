import numpy as np
import torch

from probe_transfer.alignment.methods import (
    fit_exact_permutation_alignment,
    fit_permutation_alignment,
    fit_positive_diagonal_alignment,
)
from probe_transfer.symmetry.coordinates import CoordinateTransform
from probe_transfer.symmetry.scales import seeded_positive_diagonal
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


def test_positive_diagonal_alignment_recovers_unknown_scales() -> None:
    source = np.random.default_rng(13).normal(size=(128, 16)).astype(np.float32)
    transformation = CoordinateTransform(
        "positive_diagonal", seeded_positive_diagonal(16, 42, 0.125, 8.0)
    )
    target = transformation.apply_array(source)

    fitted = fit_positive_diagonal_alignment(source, target, relative_tolerance=1e-6)

    assert fitted.scale is not None
    torch.testing.assert_close(
        fitted.scale, transformation.inverse().values.float(), rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(fitted.transform(target), source, rtol=1e-6, atol=1e-6)
