import numpy as np

from probe_transfer.alignment.methods import alignment_diagnostic, fit_ambient_alignments


def test_permutation_diagonal_recovers_known_coordinates() -> None:
    rng = np.random.default_rng(42)
    source = rng.normal(size=(512, 4)).astype(np.float32)
    indices = np.array([2, 0, 3, 1])
    scale = np.array([0.5, 1.5, 2.0, 0.75], dtype=np.float32)
    offset = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float32)
    target = np.empty_like(source)
    target[:, indices] = (source - offset) / scale

    maps = fit_ambient_alignments(
        source,
        target,
        relative_alpha=1e-6,
        shuffle_seed=314,
        device="cpu",
    )

    recovered = maps["permutation_diagonal"].transform(target)
    assert np.allclose(recovered, source, atol=1e-4)
    assert (
        alignment_diagnostic(maps["permutation_diagonal"], source, target)[
            "alignment_relative_rmse"
        ]
        < 1e-4
    )


def test_procrustes_and_ridge_generalize_better_than_shuffled_pairing() -> None:
    rng = np.random.default_rng(137)
    target = rng.normal(size=(800, 6)).astype(np.float32)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    source = (target @ orthogonal + 0.4).astype(np.float32)
    maps = fit_ambient_alignments(
        source,
        target,
        relative_alpha=1e-6,
        shuffle_seed=314,
        device="cpu",
    )

    procrustes_error = np.mean((maps["orthogonal_procrustes"].transform(target) - source) ** 2)
    ridge_error = np.mean((maps["affine_ridge"].transform(target) - source) ** 2)
    shuffled_error = np.mean((maps["shuffled_affine_ridge"].transform(target) - source) ** 2)
    assert procrustes_error < 1e-8
    assert ridge_error < 1e-8
    assert shuffled_error > 0.1


def test_alignment_fit_skips_unselected_methods(monkeypatch) -> None:
    rng = np.random.default_rng(42)
    source = rng.normal(size=(64, 4)).astype(np.float32)
    target = rng.normal(size=(64, 4)).astype(np.float32)

    def reject_assignment(*_args, **_kwargs):
        raise AssertionError("Permutation fitting must not run.")

    monkeypatch.setattr("probe_transfer.alignment.methods.linear_sum_assignment", reject_assignment)
    maps = fit_ambient_alignments(
        source,
        target,
        relative_alpha=1e-6,
        shuffle_seed=314,
        device="cpu",
        methods=["affine_ridge", "shuffled_affine_ridge"],
    )

    assert set(maps) == {"affine_ridge", "shuffled_affine_ridge"}
