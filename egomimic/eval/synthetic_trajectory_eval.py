"""Shared trajectory export for synthetic point-flow models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SyntheticTrajectoryEval:
    """Validate and export one model's display-ready point trajectory."""

    @staticmethod
    def load_validation_data(
        path: str | Path, source_key: str, particles: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load exactly ``particles`` held-out examples or fail explicitly."""
        if particles <= 0:
            raise ValueError("particles must be positive")
        archive = np.load(Path(path), allow_pickle=False)
        indices = np.flatnonzero(archive["split"] == 1)
        if len(indices) < particles:
            raise ValueError(
                f"requested {particles} validation particles, but {path} "
                f"contains only {len(indices)}"
            )
        selected = indices[:particles]
        source = torch.from_numpy(archive[source_key][selected]).float()
        target = torch.from_numpy(archive["target_3d"][selected]).float()
        return source, target

    @staticmethod
    def symmetric_nearest_neighbor_mse(
        samples: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Symmetric point-cloud NN loss, covering precision and support recall."""
        if (
            samples.ndim != 2
            or targets.ndim != 2
            or samples.shape[1:] != targets.shape[1:]
        ):
            raise ValueError("samples and targets must have shapes [N,D] and [M,D]")
        squared_distances = torch.cdist(samples, targets).square()
        sample_to_target = squared_distances.min(dim=1).values.mean()
        target_to_sample = squared_distances.min(dim=0).values.mean()
        return 0.5 * (sample_to_target + target_to_sample)

    @staticmethod
    @torch.inference_mode()
    def evaluate(model, source: torch.Tensor, target: torch.Tensor, *, steps: int):
        points = model.trajectory(source, steps=steps)
        expected = (steps + 1, len(target), 3)
        if tuple(points.shape) != expected:
            raise RuntimeError(
                f"synthetic trajectory must have shape {expected}, got {tuple(points.shape)}"
            )
        if not bool(torch.isfinite(points).all()):
            raise RuntimeError("synthetic trajectory contains non-finite points")
        return points

    @classmethod
    def export(
        cls,
        model,
        source: torch.Tensor,
        target: torch.Tensor,
        output: str | Path,
        *,
        steps: int,
    ) -> torch.Tensor:
        points = cls.evaluate(model, source, target, steps=steps)
        times = torch.linspace(0.0, 1.0, steps + 1)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            times=times.numpy(),
            points=points.detach().cpu().numpy(),
            target=target.detach().cpu().numpy(),
        )
        return points

    @staticmethod
    def export_linear_ground_truth(
        source: torch.Tensor,
        target: torch.Tensor,
        output: str | Path,
        *,
        steps: int,
    ) -> torch.Tensor:
        if source.shape[-1] == 2:
            source = torch.nn.functional.pad(source, (0, 1))
        if source.shape != target.shape or source.shape[-1] != 3:
            raise ValueError("ground-truth source and target must share shape [N,3]")
        times = torch.linspace(0.0, 1.0, steps + 1, device=source.device)
        points = torch.stack([(1.0 - time) * source + time * target for time in times])
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            times=times.cpu().numpy(),
            points=points.detach().cpu().numpy(),
            target=target.detach().cpu().numpy(),
        )
        return points
