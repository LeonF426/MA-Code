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