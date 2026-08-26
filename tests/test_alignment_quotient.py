import numpy as np
import torch

from probe_transfer.alignment.quotient import (
    build_quotient_basis,
    fit_quotient_alignment,
    quotient_scores,
)
from probe_transfer.probes.transport import StoredProbe


def _linear_probe(name: str, weight: list[float]) -> StoredProbe:
    return StoredProbe(
        name=name,
        kind="linear",
        tensors={
            "preprocessor.mean": torch.zeros(len(weight)),
            "preprocessor.scale": torch.tensor(1.0),
            "coefficient": torch.tensor([weight]),
            "intercept": torch.tensor([0.2]),
        },
        details={},
    )


def test_quotient_ridge_recovers_probe_visible_scores() -> None:
    rng = np.random.default_rng(42)
    target = rng.normal(size=(600, 3)).astype(np.float32)
    source = np.column_stack((2 * target[:, 1], target[:, 0], -target[:, 2])).astype(np.float32)
    probe = _linear_probe("probe_a", [1.0, 0.0, 0.0])
    redundant = _linear_probe("probe_b", [2.0, 0.0, 0.0])
    basis, metadata = build_quotient_basis([probe, redundant], 1e-3)
    fitted = fit_quotient_alignment(
        source,
        target,
        basis,
        relative_alpha=1e-6,
        device="cpu",
    )

    scores = quotient_scores(probe, fitted.transform(target), basis)
    assert metadata["quotient_rank"] == 1
    assert np.allclose(scores, probe.scores(source), atol=1e-4)
