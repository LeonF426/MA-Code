"""Synthetic and standard benchmark datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset


CALIFORNIA_HOUSING_FEATURES = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)


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


def prepare_california_housing_data(
    features: Any,
    targets: Any,
    *,
    train: bool = True,
    test_fraction: float = 0.2,
    seed: int = 0,
    standardize: bool = True,
    standardize_target: bool = False,
) -> TensorDataset:
    """Split and process California Housing arrays without data leakage.

    The split is deterministic. Feature and target statistics are always fitted on
    the training partition, including when the test partition is requested.
    """

    inputs = torch.as_tensor(features, dtype=torch.float32).clone()
    labels = torch.as_tensor(targets, dtype=torch.float32).reshape(-1).clone()
    if inputs.ndim != 2 or inputs.shape[1] != len(CALIFORNIA_HOUSING_FEATURES):
        raise ValueError(
            "California Housing features must have shape [n_samples, 8]"
        )
    if labels.shape[0] != inputs.shape[0]:
        raise ValueError("Features and targets must contain the same number of samples")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if inputs.shape[0] < 2:
        raise ValueError("At least two samples are required for a train/test split")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(inputs.shape[0], generator=generator)
    test_count = max(1, int(round(test_fraction * inputs.shape[0])))
    test_count = min(test_count, inputs.shape[0] - 1)
    test_indices = permutation[:test_count]
    train_indices = permutation[test_count:]

    if standardize:
        feature_mean = inputs[train_indices].mean(dim=0)
        feature_std = inputs[train_indices].std(dim=0, unbiased=False).clamp_min(1e-12)
        inputs = (inputs - feature_mean) / feature_std
    if standardize_target:
        target_mean = labels[train_indices].mean()
        target_std = labels[train_indices].std(unbiased=False).clamp_min(1e-12)
        labels = (labels - target_mean) / target_std

    selected = train_indices if train else test_indices
    return TensorDataset(inputs[selected], labels[selected])


def make_california_housing_dataset(
    *,
    root: str | Path = "data",
    train: bool = True,
    test_fraction: float = 0.2,
    seed: int = 0,
    standardize: bool = True,
    standardize_target: bool = False,
    download: bool = True,
) -> TensorDataset:
    """Load sklearn's California Housing regression benchmark as tensors.

    The first call downloads approximately 1.5 MB when ``download=True``;
    subsequent calls use sklearn's cache under ``root``.
    """

    try:
        from sklearn.datasets import fetch_california_housing
    except ImportError as exc:
        raise ImportError(
            "California Housing requires `pip install -e '.[tabular]'`."
        ) from exc

    bunch = fetch_california_housing(
        data_home=str(Path(root)),
        download_if_missing=download,
    )
    return prepare_california_housing_data(
        bunch.data,
        bunch.target,
        train=train,
        test_fraction=test_fraction,
        seed=seed,
        standardize=standardize,
        standardize_target=standardize_target,
    )


def build_dataset(config: Mapping[str, Any], train: bool = True) -> Dataset:
    """Build a synthetic, tabular, or torchvision dataset from a dictionary."""

    name = str(config.get("name", "linear_regression")).lower()
    if name == "linear_regression":
        return make_linear_regression_dataset(
            n_samples=int(config.get("n_samples", 1024)),
            input_dim=int(config["input_dim"]),
            target_weights=config.get("target_weights"),
            noise_std=float(config.get("noise_std", 0.0)),
            seed=int(config.get("seed", 0)) + (0 if train else 1),
        )
    if name in {"california_housing", "california"}:
        return make_california_housing_dataset(
            root=Path(config.get("root", "data")),
            train=train,
            test_fraction=float(
                config.get("test_fraction", config.get("test_size", 0.2))
            ),
            seed=int(config.get("seed", 0)),
            standardize=bool(config.get("standardize", True)),
            standardize_target=bool(config.get("standardize_target", False)),
            download=bool(config.get("download", True)),
        )

    torchvision_names = {
        "mnist": "MNIST",
        "fashion_mnist": "FashionMNIST",
        "cifar10": "CIFAR10",
        "cifar100": "CIFAR100",
    }
    if name not in torchvision_names:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: linear_regression, "
            f"california_housing, {sorted(torchvision_names)}"
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

