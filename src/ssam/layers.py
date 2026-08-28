"""Small reusable neural-network layers."""

from __future__ import annotations

import torch
from torch import nn


class DiagLinear(nn.Module):
    """A memory-efficient square linear layer with a diagonal weight matrix."""

    def __init__(self, dim: int, bias: bool = False) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.empty(dim)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs * self.weight
        return outputs if self.bias is None else outputs + self.bias

    def extra_repr(self) -> str:
        return f"dim={self.dim}, bias={self.bias is not None}"
