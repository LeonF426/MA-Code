# src/lin_sgd/layers.py
import torch
import torch.nn as nn
import math

class DiagLinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(1.0)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight
