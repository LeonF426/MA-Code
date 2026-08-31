"""Small reusable neural-network layers."""
from __future__ import annotations
import math

import torch
from torch import nn


class DiagLinear(nn.Module):
    def __init__(
        self,
        dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if dim < 1:
            raise ValueError("dim must be positive")

        self.dim = dim
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = (
            nn.Parameter(torch.empty(dim))
            if bias
            else None
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(
            self.weight,
            -bound,
            bound,
        )

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        outputs = inputs * self.weight

        if self.bias is not None:
            outputs = outputs + self.bias

        return outputs
