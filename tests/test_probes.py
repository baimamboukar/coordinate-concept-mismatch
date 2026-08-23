import numpy as np

from probe_transfer.probes import CPDegree2, FrozenPreprocessor, train_linear_probe


def test_frozen_preprocessor_centers_training_activations() -> None:
    activations = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)

    transformed = FrozenPreprocessor.fit(activations).transform(activations)

    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6)


def test_linear_probe_selects_on_source_validation() -> None:
    train_x = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    train_y = np.array([0, 0, 0, 1, 1, 1])
    validation_x = np.array([[-4.0], [-0.5], [0.5], [4.0]])
    validation_y = np.array([0, 0, 1, 1])

    probe = train_linear_probe(
        train_x,
        train_y,
        validation_x,
        validation_y,
        c_values=[0.1, 1.0],
    )

    assert probe.validation_auroc == 1.0
    assert probe.scores(validation_x).shape == (4,)


def test_cp_parameter_count_matches_affine_completed_formula() -> None:
    input_size = 8
    rank = 2
    model = CPDegree2(input_size, rank)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 2 * rank * (input_size + 1) + rank + (input_size + 1)
