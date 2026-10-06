"""Deterministic, matched mini-batches for fixed PINN collocation sets."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch


class _PermutationStream:
    """Yield each point once per shuffled pass, like a non-dropping DataLoader."""

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size < 1:
            raise ValueError("PINN point groups must not be empty")
        if batch_size < 1 or batch_size > size:
            raise ValueError("PINN batch sizes must lie between one and the group size")
        self.size = size
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self._permutation = torch.empty(0, dtype=torch.long)
        self._cursor = size

    def next_indices(self) -> torch.Tensor:
        if self.batch_size == self.size:
            return torch.arange(self.size)
        if self._cursor >= self.size:
            self._permutation = torch.randperm(
                self.size,
                generator=self.generator,
            )
            self._cursor = 0
        end = min(self._cursor + self.batch_size, self.size)
        indices = self._permutation[self._cursor:end]
        self._cursor = end
        return indices


class PINNBatcher:
    """Create reproducible, ratio-preserving batches for PINN point groups.

    ``training.batch_size`` is the interior-point batch size. Other point groups
    receive the same fraction of their full set. For example, 128/512 interior
    points implies 64/256 boundary points. ``gd`` always uses every point.
    """

    def __init__(
        self,
        point_groups: Mapping[str, torch.Tensor],
        training: Mapping[str, Any],
    ) -> None:
        if "interior" not in point_groups:
            raise ValueError("PINN batching requires an 'interior' point group")
        if not point_groups:
            raise ValueError("PINN batching requires at least one point group")

        self.point_groups = dict(point_groups)
        algorithm = str(training["algorithm"]).lower().replace("-", "_")
        interior_size = int(self.point_groups["interior"].shape[0])
        requested = int(training.get("batch_size", interior_size))
        if requested < 1:
            raise ValueError("training.batch_size must be positive")
        fraction = 1.0 if algorithm == "gd" else min(1.0, requested / interior_size)
        base_seed = int(training.get("batch_seed", training.get("seed", 0)))

        self.batch_sizes: dict[str, int] = {}
        self._streams: dict[str, _PermutationStream] = {}
        for index, (name, points) in enumerate(self.point_groups.items()):
            size = int(points.shape[0])
            if size < 1:
                raise ValueError(f"PINN point group {name!r} must not be empty")
            batch_size = size if algorithm == "gd" else max(1, math.ceil(fraction * size))
            batch_size = min(size, batch_size)
            self.batch_sizes[name] = batch_size
            # Stable arithmetic offsets avoid Python's process-randomized hash().
            seed = (base_seed + 104_729 * index) % (2**63 - 1)
            self._streams[name] = _PermutationStream(size, batch_size, seed)

    def next(self) -> dict[str, torch.Tensor]:
        batches: dict[str, torch.Tensor] = {}
        for name, points in self.point_groups.items():
            indices = self._streams[name].next_indices().to(points.device)
            batches[name] = points.index_select(0, indices)
        return batches

