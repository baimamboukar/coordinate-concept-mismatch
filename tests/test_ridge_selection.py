from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge

from core.config import ConfigError, load_config
from core.constants import CONFIGS_DIR
from pipeline.config import materialize_stage
from pipeline.panel import select_task
from probe_transfer.alignment.methods import fit_affine_ridge
from probe_transfer.alignment.ridge import RidgeSystem
from probe_transfer.alignment.selection import (
    fit_configured_alignments,
    fit_grouped_ridge,
    validate_alignment_selection,
)


def test_weighted_ridge_matches_independent_reference() -> None:
    rng = np.random.default_rng(42)
    target, source = rng.normal(size=(80, 5)), rng.normal(size=(80, 3))
    weights = rng.uniform(0.1, 3.0, 80)
    system = RidgeSystem.prepare(
        torch.from_numpy(source), torch.from_numpy(target), torch.from_numpy(weights)
    )
    weight, bias, penalty = system.solve(0.02)
    reference = Ridge(alpha=penalty, solver="svd").fit(
        target, source, sample_weight=weights / weights.mean()
    )
    np.testing.assert_allclose(weight.numpy(), reference.coef_.T, atol=1e-12)
    np.testing.assert_allclose(bias.numpy(), reference.intercept_, atol=1e-12)
    scaled = RidgeSystem.prepare(
        torch.from_numpy(source), torch.from_numpy(target), torch.from_numpy(weights * 100)
    ).solve(0.02)
    torch.testing.assert_close(weight, scaled[0])


@pytest.mark.parametrize(
    "weights",
    [
        torch.ones(3),
        torch.tensor([1.0, 0.0, 1.0, 1.0]),
        torch.tensor([1.0, float("nan"), 1.0, 1.0]),
    ],
)
def test_invalid_weights_fail_before_solving(weights) -> None:
    with pytest.raises(ValueError, match="sample weights"):
        RidgeSystem.prepare(torch.ones(4, 2), torch.ones(4, 2), weights)


def _data():
    rng = np.random.default_rng(137)
    target = rng.normal(size=(120, 4)).astype(np.float32)
    source = target @ np.diag([2.0, -1.0, 0.5, 1.0]).astype(np.float32) + 0.2
    source[60:] *= 10
    target[60:] *= 10
    validation = {}
    for name, scale in (("small", 1), ("large", 10)):
        values = rng.normal(size=(24, 4)).astype(np.float32)
        expected = values @ np.diag([2.0, -1.0, 0.5, 1.0]).astype(np.float32) + 0.2
        validation[name] = (expected * scale, values * scale)
    return source, target, {"small": 60, "large": 60}, validation


def test_uniform_fixed_preserves_the_original_estimator() -> None:
    source, target, sizes, validation = _data()
    maps, records = fit_grouped_ridge(
        source,
        target,
        sizes,
        validation,
        weighting="uniform",
        relative_alphas=[1e-4],
        shuffle_seed=42,
        device="cpu",
    )
    reference = fit_affine_ridge(
        torch.from_numpy(source),
        torch.from_numpy(target),
        relative_alpha=1e-4,
        method="affine_ridge",
    )
    np.testing.assert_array_equal(
        maps["affine_ridge"].transform(target), reference.transform(target)
    )
    assert len(records) == 4
    assert all(row["selected"] for row in records)


def test_selection_minimizes_worst_task_error_and_retains_all_candidates() -> None:
    maps, records = fit_grouped_ridge(
        *_data(),
        weighting="source_variance",
        relative_alphas=[1e-6, 1e-4, 1e-2],
        shuffle_seed=42,
        device="cpu",
    )
    assert len(records) == 12
    for method, fitted in maps.items():
        candidates = [row for row in records if row["method"] == method]
        best = min(
            candidates, key=lambda row: (row["selection_max_relative_mse"], -row["relative_alpha"])
        )
        assert fitted.metadata["ridge_relative_alpha"] == best["relative_alpha"]
        assert sum(row["selected"] for row in candidates) == 2
        weights = {row["fit_task"]: row["sample_weight"] for row in candidates}
        assert weights["small"] > 50 * weights["large"]
        for alpha in (1e-6, 1e-4, 1e-2):
            rows = [row for row in candidates if row["relative_alpha"] == alpha]
            assert sum(row["weighted_train_loss_fraction"] for row in rows) == pytest.approx(1)
            assert rows[0]["selection_max_relative_mse"] == max(
                row["validation_relative_mse"] for row in rows
            )


def test_shuffle_preserves_each_task_marginal(monkeypatch) -> None:
    source, target, sizes, validation = _data()
    original = RidgeSystem.prepare.__func__
    observed = []

    def capture(cls, values, inputs, weights=None):
        observed.append(values.clone())
        return original(cls, values, inputs, weights)

    monkeypatch.setattr(RidgeSystem, "prepare", classmethod(capture))
    fit_grouped_ridge(
        source,
        target,
        sizes,
        validation,
        weighting="uniform",
        relative_alphas=[1e-4],
        shuffle_seed=42,
        device="cpu",
    )
    assert not torch.equal(observed[0], observed[1])
    for first, second in zip(observed[0].split(60), observed[1].split(60), strict=True):
        torch.testing.assert_close(first.sort(dim=0).values, second.sort(dim=0).values)


def test_selection_is_deterministic() -> None:
    first = fit_grouped_ridge(
        *_data(),
        weighting="source_variance",
        relative_alphas=[1e-4, 1e-2],
        shuffle_seed=42,
        device="cpu",
    )
    second = fit_grouped_ridge(
        *_data(),
        weighting="source_variance",
        relative_alphas=[1e-4, 1e-2],
        shuffle_seed=42,
        device="cpu",
    )
    assert first[1] == second[1]


def test_selection_loads_only_fitting_task_validation(tmp_path: Path, monkeypatch) -> None:
    study = load_config(CONFIGS_DIR / "studies/smollm_shared_map_compatibility.yaml")
    config = materialize_stage(select_task(study, "ag_news", "uniform_fixed"), "align")
    source, target, _, validation = _data()
    calls = []
    for entry in config["fit_materials"]["datasets"]:
        entry.update(fit_rows=60, expected_validation_rows=24)

    def paired(root, _source, _target, split, layer):
        calls.append((root.name, split, layer))
        values = validation["small" if root.name.startswith("sst2") else "large"]
        return *values, np.arange(24), np.zeros(24)

    monkeypatch.setattr("probe_transfer.alignment.selection.paired_split", paired)
    _, records = fit_configured_alignments(
        source,
        target,
        config,
        tmp_path,
        source="smollm1",
        target="smollm2",
        data_seed=42,
        layer="layer_75",
        shuffle_seed=42,
        device="cpu",
    )
    assert len(records) == 4
    assert calls == [
        ("sst2-sentiment-v1", "seed_42_validation", "layer_75"),
        ("wildguardmix-v1", "seed_42_validation", "layer_75"),
    ]


@pytest.mark.parametrize(
    "settings",
    [
        {"weighting": "labels", "relative_alphas": [1e-4]},
        {"weighting": "uniform", "relative_alphas": [float("nan")]},
        {"weighting": "uniform", "relative_alphas": [1e-4, 1e-6]},
        {"weighting": "uniform", "relative_alphas": [1e-4], "split": "test"},
    ],
)
def test_invalid_selection_configuration_is_rejected(settings) -> None:
    study = load_config(CONFIGS_DIR / "studies/smollm_shared_map_compatibility.yaml")
    config = materialize_stage(select_task(study, "sst2", "uniform_fixed"), "align")
    config["alignment"]["fitting"] = settings
    with pytest.raises(ConfigError):
        validate_alignment_selection(config)
