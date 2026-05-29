# src/lin_sgd/data.py
import torch
from typing import Callable, Tuple, Optional
from pathlib import Path

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

def create_fixed_dataset(
    n_samples: int,
    d: int,
    w_star: torch.Tensor,
    noise_std: float = 0.0,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a fixed dataset (X, Y) and optionally save to disk.
    
    X ~ N(0, I_d)
    Y = w_star^T X + eps,  eps ~ N(0, noise_std^2)
    
    Returns:
        X: (n_samples, d)
        Y: (n_samples,)
    """
    X = torch.randn(n_samples, d, device=device)
    eps = noise_std * torch.randn(n_samples, device=device)
    Y = X @ w_star + eps
    
    if save_path is not None:
        data = {"X": X.cpu(), "Y": Y.cpu(), "w_star": w_star.cpu(), "noise_std": noise_std}
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)
        print(f"Saved dataset to {save_path}")   
    return X, Y

def make_batch_sampler_from_fixed_dataset(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a callable that samples random minibatches from a fixed dataset.
    
    X: (n_samples, d)
    Y: (n_samples,)
    
    Returns:
        sampler(batch_size) -> (X_batch, Y_batch)
    """
    n_samples = X.shape[0]
    
    def sampler(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        #indices = torch.randint(0, n_samples, (batch_size,), device=X.device)
        return X, Y
    
    return sampler


def load_fixed_dataset(
    load_path: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Load a previously saved dataset.
    
    Returns:
        X: (n_samples, d)
        Y: (n_samples,)
        metadata: dict with w_star, noise_std, etc.
    """
    data = torch.load(load_path, map_location=device)
    X = data["X"].to(device)
    Y = data["Y"].to(device)
    metadata = {k: v for k, v in data.items() if k not in ["X", "Y"]}
    print(f"Loaded dataset from {load_path}, X.shape={X.shape}, Y.shape={Y.shape}")
    return X, Y, metadata

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d = 10
    n_train = 10000
    n_test = 2000
    w_star = torch.randn(d, device=device)
    noise_std = 0.1
    
    # create training set
    X_train, Y_train = create_fixed_dataset(
        n_samples=n_train,
        d=d,
        w_star=w_star,
        noise_std=noise_std,
        device=device,
        save_path=f"data/train_dataset_d{d}.pt",
    )
    
    # create test set (same w_star)
    X_test, Y_test = create_fixed_dataset(
        n_samples=n_test,
        d=d,
        w_star=w_star,
        noise_std=noise_std,
        device=device,
        save_path=f"data/test_dataset_d{d}.pt",
    )

if __name__ == "__main__":
    main()
