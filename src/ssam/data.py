"""Synthetic and standard benchmark datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset


def make_linear_regression_dataset(
    n_samples: int,
    input_dim: int,
    target_weights: Sequence[float] | torch.Tensor | None = None,
    noise_std: float = 0.0,
    seed: int = 0,
) -> TensorDataset:
    """Create a reproducible Gaussian linear-regression dataset."""

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n_samples, input_dim, generator=generator)
    weights = (
        torch.randn(input_dim, generator=generator)
        if target_weights is None
        else torch.as_tensor(target_weights, dtype=inputs.dtype)
    )
    if weights.shape != (input_dim,):
        raise ValueError("target_weights must have shape [input_dim]")
    targets = inputs @ weights
    if noise_std:
        targets = targets + noise_std * torch.randn(n_samples, generator=generator)
    return TensorDataset(inputs, targets)


def build_dataset(config: Mapping[str, Any], train: bool = True) -> Dataset:
    """Build a synthetic dataset or a torchvision benchmark from a dictionary."""

    name = str(config.get("name", "linear_regression")).lower()
    if name == "linear_regression":
        return make_linear_regression_dataset(
            n_samples=int(config.get("n_samples", 1024)),
            input_dim=int(config["input_dim"]),
            target_weights=config.get("target_weights"),
            noise_std=float(config.get("noise_std", 0.0)),
            seed=int(config.get("seed", 0)) + (0 if train else 1),
        )

    torchvision_names = {
        "mnist": "MNIST",
        "fashion_mnist": "FashionMNIST",
        "cifar10": "CIFAR10",
        "cifar100": "CIFAR100",
    }
    if name not in torchvision_names:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: linear_regression, {sorted(torchvision_names)}"
        )
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "Torchvision datasets require `pip install -e '.[benchmarks]'`."
        ) from exc

    image_size = int(config.get("image_size", 224))
    channels = 1 if name in {"mnist", "fashion_mnist"} else 3
    transform_steps: list[Any] = [transforms.Resize((image_size, image_size))]
    if channels == 1 and bool(config.get("rgb", True)):
        transform_steps.append(transforms.Grayscale(num_output_channels=3))
    transform_steps.append(transforms.ToTensor())
    dataset_class = getattr(datasets, torchvision_names[name])
    return dataset_class(
        root=str(Path(config.get("root", "data"))),
        train=train,
        download=bool(config.get("download", False)),
        transform=transforms.Compose(transform_steps),
    )
