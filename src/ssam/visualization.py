"""Visual diagnostics for histories, checkpoint trajectories, and loss slices."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .trainers import TrainingResult


def _pyplot(show: bool = False):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires `pip install -e '.[visualization]'`.") from exc
    return plt


def _finish(fig, path: str | Path | None, show: bool):
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160, bbox_inches="tight")
    if show:
        fig.show()
    return fig


def plot_training_history(
    result: TrainingResult | dict[str, list[float | int]],
    path: str | Path | None = None,
    show: bool = False,
):
    """Plot loss, balancedness, learning-rate, and sharpness histories."""

    plt = _pyplot(show)
    history = result.history if isinstance(result, TrainingResult) else result
    steps = history["step"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    series = [
        ("loss", "Loss"),
        ("layer_balance", "Layer Balance"),
        ("learning_rate", "Learning rate"),
        ("sharpness_scale", "Sharpness scale"),
    ]
    for axis, (key, label) in zip(axes, series):
        if key == "layer_balance":
            if history[key] and history[key][0]:
                labels = [
                    f"layers {i + 2} & {i + 1}"
                    for i in range(len(history[key][0]))
                ]
                axis.plot(steps, history[key], label=labels)
                axis.legend()
        elif key == "loss" and "clean_loss" in history:
            axis.plot(steps, history["clean_loss"], label="clean")
            regularized = np.asarray(history.get("regularized_loss", []), dtype=float)
            if regularized.size and np.isfinite(regularized).any():
                axis.plot(steps, regularized, label="regularized estimate", alpha=0.8)
            axis.legend()
        else:
            axis.plot(steps, history[key])
        axis.set(xlabel="Step", ylabel=label, title=label)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return _finish(fig, path, show)


def _pca(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    embedding = left[:, :2] * singular[:2]
    if embedding.shape[1] == 1:
        embedding = np.column_stack([embedding, np.zeros(len(embedding))])
    return embedding


def plot_checkpoint_embedding(
    result: TrainingResult,
    method: str = "tsne",
    path: str | Path | None = None,
    show: bool = False,
    random_state: int = 0,
):
    """Embed checkpoint parameters in 2D and color them by observed loss.

    PCA preserves a global linear projection. t-SNE is useful for finding groups
    of similar checkpoints, but should not be interpreted as metric geometry.
    """

    if len(result.parameter_snapshots) < 3:
        raise ValueError("At least three checkpoints are required; lower checkpoint_every")
    values = torch.stack(result.parameter_snapshots).numpy()
    method = method.lower()
    if method == "pca":
        embedding = _pca(values)
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError(
                "t-SNE requires `pip install -e '.[visualization]'`. Use method='pca' without it."
            ) from exc
        perplexity = min(30.0, max(2.0, len(values) / 3), len(values) - 1.0)
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca" if values.shape[1] >= 2 else "random",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(values)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    losses = np.asarray(result.snapshot_losses, dtype=float)
    if np.isnan(losses[0]):
        losses[0] = losses[1] if len(losses) > 1 else 0.0
    plt = _pyplot(show)
    fig, axis = plt.subplots(figsize=(6.5, 5))
    points = axis.scatter(embedding[:, 0], embedding[:, 1], c=losses, cmap="viridis", s=38)
    axis.plot(embedding[:, 0], embedding[:, 1], color="0.6", linewidth=0.8, alpha=0.7)
    axis.scatter(*embedding[0], marker="s", label="start", color="tab:blue")
    axis.scatter(*embedding[-1], marker="*", s=120, label="end", color="tab:red")
    axis.set(title=f"Checkpoint trajectory ({method.upper()})", xlabel="Component 1", ylabel="Component 2")
    axis.legend()
    fig.colorbar(points, ax=axis, label="Training loss")
    fig.tight_layout()
    return _finish(fig, path, show)


def _random_direction(parameters: list[nn.Parameter], generator: torch.Generator) -> list[torch.Tensor]:
    direction = [torch.randn(parameter.shape, generator=generator, device="cpu").to(parameter.device) for parameter in parameters]
    norm = torch.sqrt(sum(value.square().sum() for value in direction)).clamp_min(1e-12)
    return [value / norm for value in direction]


def plot_loss_landscape(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    radius: float = 1.0,
    resolution: int = 21,
    path: str | Path | None = None,
    show: bool = False,
    seed: int = 0,
):
    """Evaluate a true 2D random parameter slice around the current model."""

    if resolution < 3:
        raise ValueError("resolution must be at least 3")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    center = [parameter.detach().clone() for parameter in parameters]
    generator = torch.Generator().manual_seed(seed)
    first = _random_direction(parameters, generator)
    second = _random_direction(parameters, generator)
    projection = sum((a * b).sum() for a, b in zip(first, second))
    second = [b - projection * a for a, b in zip(first, second)]
    norm = torch.sqrt(sum(value.square().sum() for value in second)).clamp_min(1e-12)
    second = [value / norm for value in second]

    device = center[0].device
    inputs, targets = inputs.to(device), targets.to(device)
    coordinates = np.linspace(-radius, radius, resolution)
    losses = np.empty((resolution, resolution), dtype=float)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row, y_value in enumerate(coordinates):
                for column, x_value in enumerate(coordinates):
                    for parameter, base, x_direction, y_direction in zip(parameters, center, first, second):
                        parameter.copy_(base + x_value * x_direction + y_value * y_direction)
                    losses[row, column] = float(loss_fn(model(inputs), targets))
    finally:
        with torch.no_grad():
            for parameter, base in zip(parameters, center):
                parameter.copy_(base)
        model.train(was_training)

    plt = _pyplot(show)
    fig, axis = plt.subplots(figsize=(6.5, 5.2))
    contour = axis.contourf(coordinates, coordinates, losses, levels=30, cmap="viridis")
    axis.scatter([0], [0], marker="*", s=110, color="white", edgecolor="black", label="trained parameters")
    axis.set(title="Random 2D loss slice", xlabel="Direction 1", ylabel="Direction 2")
    axis.legend()
    fig.colorbar(contour, ax=axis, label="Loss")
    fig.tight_layout()
    return _finish(fig, path, show)

