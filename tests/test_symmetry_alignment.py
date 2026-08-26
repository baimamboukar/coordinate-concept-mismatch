import numpy as np
import torch

from probe_transfer.alignment.methods import fit_permutation_alignment
from probe_transfer.symmetry.transforms import inverse_permutation, seeded_permutation


def test_activation_estimation_recovers_unknown_permutation() -> None:
    source = np.random.default_rng(7).normal(size=(128, 16)).astype(np.float32)
    permutation = seeded_permutation(16, 42)
    target = source[:, permutation.numpy()]

    fitted = fit_permutation_alignment(source, target, device="cpu")

    assert fitted.indices is not None
    torch.testing.assert_close(fitted.indices.cpu(), inverse_permutation(permutation))
    np.testing.assert_allclose(fitted.transform(target), source, rtol=0, atol=0)
