# src/lin_sgd/data.py
import torch
from typing import Callable, Tuple

def sample_gaussian_linear(
    n: int,
    d: int,
    device: torch.device,
    w_star: torch.Tensor = None,
    noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n, d, device=device)
    if w_star is None:
        w_star = torch.randn(d, device=device)
    eps = noise_std * torch.randn(n, device=device)
    Y = X @ w_star + eps
    return X, Y
