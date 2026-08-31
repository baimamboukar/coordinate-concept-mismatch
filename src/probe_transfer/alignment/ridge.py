from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class RidgeSystem:
    gram: torch.Tensor
    cross: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor

    @classmethod
    def prepare(
        cls,
        source: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> "RidgeSystem":
        if source.ndim != 2 or target.ndim != 2 or len(source) != len(target) or len(source) < 2:
            raise ValueError("Ridge fitting requires paired two-dimensional arrays.")
        if weights is None:
            source_mean, target_mean = source.mean(dim=0), target.mean(dim=0)
            centered_source, centered_target = source - source_mean, target - target_mean
            left = centered_target.T
        else:
            weights = weights.to(device=target.device, dtype=target.dtype)
            if (
                weights.shape != (len(target),)
                or not torch.isfinite(weights).all()
                or (weights <= 0).any()
            ):
                raise ValueError("Ridge sample weights must be finite, positive, and row-aligned.")
            weights = weights / weights.mean()
            source_mean = (source * weights[:, None]).mean(dim=0)
            target_mean = (target * weights[:, None]).mean(dim=0)
            centered_source, centered_target = source - source_mean, target - target_mean
            left = (centered_target * weights[:, None]).T
        return cls(left @ centered_target, left @ centered_source, source_mean, target_mean)

    def solve(self, relative_alpha: float) -> tuple[torch.Tensor, torch.Tensor, float]:
        if not isfinite(relative_alpha) or relative_alpha <= 0:
            raise ValueError("Relative Ridge alpha must be finite and positive.")
        scale = self.gram.diagonal().mean().clamp_min(torch.finfo(self.gram.dtype).eps)
        penalty = relative_alpha * scale
        regularized = self.gram + penalty * torch.eye(
            self.gram.shape[0], dtype=self.gram.dtype, device=self.gram.device
        )
        weight = torch.linalg.solve(regularized, self.cross)
        bias = self.source_mean - self.target_mean @ weight
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("Ridge fitting produced a non-finite map.")
        return weight, bias, float(penalty.item())
