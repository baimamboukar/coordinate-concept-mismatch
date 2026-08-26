from pathlib import Path

import numpy as np
import pytest
import torch

from probe_transfer.probes.transport import StoredProbe, load_probe_bundle, save_probe_bundle
from probe_transfer.symmetry.transforms import seeded_permutation


def probes(width: int) -> dict[str, StoredProbe]:
    generator = torch.Generator().manual_seed(9)
    common = {
        "preprocessor.mean": torch.randn(width, generator=generator),
        "preprocessor.scale": torch.tensor(1.7),
    }
    return {
        "layer_75.linear": StoredProbe(
            "layer_75.linear",
            "linear",
            {
                **common,
                "coefficient": torch.randn(1, width, generator=generator),
                "intercept": torch.randn(1, generator=generator),
            },
            {"kind": "linear"},
        ),
        "layer_75.cp_degree_2": StoredProbe(
            "layer_75.cp_degree_2",
            "CPDegree2",
            {
                **common,
                "model.left.weight": torch.randn(3, width, generator=generator),
                "model.left.bias": torch.randn(3, generator=generator),
                "model.right.weight": torch.randn(3, width, generator=generator),
                "model.right.bias": torch.randn(3, generator=generator),
                "model.alpha": torch.randn(3, generator=generator),
                "model.linear.weight": torch.randn(1, width, generator=generator),
                "model.linear.bias": torch.randn(1, generator=generator),
            },
            {"kind": "CPDegree2"},
        ),
        "layer_75.mlp": StoredProbe(
            "layer_75.mlp",
            "OneHiddenLayerMLP",
            {
                **common,
                "model.network.0.weight": torch.randn(4, width, generator=generator),
                "model.network.0.bias": torch.randn(4, generator=generator),
                "model.network.2.weight": torch.randn(1, 4, generator=generator),
                "model.network.2.bias": torch.randn(1, generator=generator),
            },
            {"kind": "OneHiddenLayerMLP"},
        ),
    }


@pytest.mark.parametrize("name", ["layer_75.linear", "layer_75.cp_degree_2", "layer_75.mlp"])
def test_exact_probe_transport_preserves_scores(name: str) -> None:
    activations = np.random.default_rng(3).normal(size=(12, 8)).astype(np.float32)
    permutation = seeded_permutation(8, 42)
    probe = probes(8)[name]

    expected = probe.scores(activations)
    actual = probe.transport(permutation).scores(activations[:, permutation])

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_transported_bundle_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "transported.safetensors"
    permutation = seeded_permutation(8, 137)
    transported = {name: probe.transport(permutation) for name, probe in probes(8).items()}

    digest = save_probe_bundle(path, transported, metadata_updates={"permutation_seed": 137})
    loaded = load_probe_bundle(path)

    assert set(loaded) == set(transported)
    assert loaded["layer_75.linear"].details["permutation_seed"] == 137
    assert len(digest) == 64
