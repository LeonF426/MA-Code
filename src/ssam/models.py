# src/lin_sgd/models.py
import torch
import torch.nn as nn
from typing import List, Dict
from .layers import DiagLinear

class MixedLinearNet(nn.Module):
    def __init__(self, layer_specs: List[Dict]):
        super().__init__()
        layers = []
        for spec in layer_specs:
            t = spec["type"]
            m, n = spec["in_dim"], spec["out_dim"]
            if t == "diag":
                assert m == n
                layer = DiagLinear(m)
            elif t == "dense":
                layer = nn.Linear(m, n, bias=False)
            else:
                raise ValueError(f"Unknown layer type {t}")
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            # activation hook (identity for now)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        return h.sum(dim=1)
