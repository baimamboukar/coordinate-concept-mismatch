from dataclasses import dataclass
from typing import Any

from probe_transfer.symmetry.coordinates import CoordinateTransform
from probe_transfer.symmetry.scales import seeded_positive_diagonal
from probe_transfer.symmetry.transforms import seeded_gqa_head_permutation, seeded_permutation


@dataclass(frozen=True)
class TransformationCase:
    seed: int
    coordinates: CoordinateTransform
    variant: str | None = None
    scale_minimum: float | None = None
    scale_maximum: float | None = None

    @property
    def key(self) -> str:
        parts = [self.coordinates.artifact_label]
        if self.variant is not None:
            parts.append(self.variant)
        parts.append(str(self.seed))
        return "_".join(parts)

    @property
    def map_key(self) -> str:
        return str(self.seed) if self.variant is None else f"{self.variant}:{self.seed}"

    def fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"transformation_seed": self.seed}
        if self.variant is not None:
            fields["transformation_variant"] = self.variant
        if self.scale_minimum is not None and self.scale_maximum is not None:
            fields.update(
                {
                    "scale_minimum": self.scale_minimum,
                    "scale_maximum": self.scale_maximum,
                }
            )
        return fields


def build_transformation_cases(symmetry: dict[str, Any]) -> tuple[TransformationCase, ...]:
    seeds = symmetry["transformation_seeds"]
    transformation = symmetry["transformation"]
    if transformation == "mlp_positive_diagonal":
        ranges = symmetry.get("scale_ranges")
        configured = (
            [(variant, values) for variant, values in ranges.items()]
            if ranges is not None
            else [(None, symmetry["scale_range"])]
        )
        return tuple(
            TransformationCase(
                seed=seed,
                coordinates=CoordinateTransform(
                    "positive_diagonal",
                    seeded_positive_diagonal(symmetry["width"], seed, minimum, maximum),
                ),
                variant=variant,
                scale_minimum=minimum,
                scale_maximum=maximum,
            )
            for variant, (minimum, maximum) in configured
            for seed in seeds
        )
    if transformation != "attention_head_permutation":
        return tuple(
            TransformationCase(
                seed=seed,
                coordinates=CoordinateTransform(
                    "permutation", seeded_permutation(symmetry["width"], seed)
                ),
            )
            for seed in seeds
        )
    layout = symmetry["attention_layout"]
    return tuple(
        TransformationCase(
            seed=seed,
            coordinates=CoordinateTransform(
                "permutation",
                seeded_gqa_head_permutation(
                    layout["query_heads"],
                    layout["key_value_heads"],
                    layout["head_dim"],
                    seed,
                ),
            ),
        )
        for seed in seeds
    )
