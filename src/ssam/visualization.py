"""Visual diagnostics for histories, checkpoint trajectories, and loss slices."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .average_sharpness import AverageSharpnessInterpolationResult
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
    steps = history.get("step", list(range(len(history.get("loss", [])))))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    series = [
        ("loss", "Loss"),
        ("layer_balance", "Layer Balance"),
        ("learning_rate", "Learning rate"),
        ("sharpness_scale", "Sharpness scale"),
    ]
    for axis, (key, label) in zip(axes, series):
        if key == "layer_balance":
            values = history.get(key, [])
            if values and values[0]:
                labels = [
                    f"layers {i + 2} & {i + 1}"
                    for i in range(len(values[0]))
                ]
                axis.plot(steps, values, label=labels)
                axis.legend()
        elif key == "loss" and "clean_loss" in history:
            clean = np.asarray(history.get("clean_loss", []), dtype=float)
            if clean.size and np.isfinite(clean).any():
                axis.plot(steps[: clean.size], clean, label="clean")
            regularized = np.asarray(history.get("regularized_loss", []), dtype=float)
            if regularized.size and np.isfinite(regularized).any():
                axis.plot(
                    steps[: regularized.size],
                    regularized,
                    label="regularized estimate",
                    alpha=0.8,
                )
            if axis.lines:
                axis.legend()
        else:
            values = history.get(key, [])
            if values:
                axis.plot(steps[: len(values)], values)
        axis.set(xlabel="Step", ylabel=label, title=label)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_sharpness_interpolation(
    result: AverageSharpnessInterpolationResult,
    path: str | Path | None = None,
    show: bool = False,
    endpoint_labels: tuple[str, str] = ("start", "end"),
):
    """Plot clean loss and Gaussian sharpness along a parameter interpolation."""

    if len(endpoint_labels) != 2:
        raise ValueError("endpoint_labels must contain exactly two labels")
    if not result.points:
        raise ValueError("The interpolation result contains no points")

    plt = _pyplot(show)
    coefficients = np.asarray(
        [point.coefficient for point in result.points], dtype=float
    )
    clean_loss = np.asarray(
        [point.clean_loss for point in result.points], dtype=float
    )
    regularized_loss = np.asarray(
        [point.regularized_loss for point in result.points], dtype=float
    )
    sharpness = np.asarray(
        [point.average_sharpness for point in result.points], dtype=float
    )
    confidence_low = np.asarray(
        [point.confidence_low for point in result.points], dtype=float
    )
    confidence_high = np.asarray(
        [point.confidence_high for point in result.points], dtype=float
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(coefficients, clean_loss, marker="o", label="clean loss")
    axes[0].plot(
        coefficients,
        regularized_loss,
        marker="o",
        linestyle="--",
        label="Gaussian mean loss",
    )
    positive_losses = np.concatenate((clean_loss, regularized_loss))
    if positive_losses.size and np.all(positive_losses > 0.0):
        axes[0].set_yscale("log")
    axes[0].set_title("Loss along interpolation")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(coefficients, sharpness, marker="o", label="average sharpness")
    finite_band = np.isfinite(confidence_low) & np.isfinite(confidence_high)
    if finite_band.any():
        axes[1].fill_between(
            coefficients[finite_band],
            confidence_low[finite_band],
            confidence_high[finite_band],
            alpha=0.2,
            label="confidence interval",
        )
    axes[1].set_title("Gaussian sharpness along interpolation")
    axes[1].set_ylabel("E[L(theta + noise)] - L(theta)")
    axes[1].legend()

    x_label = f"t  (0 = {endpoint_labels[0]}, 1 = {endpoint_labels[1]})"
    for axis in axes:
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_pinn_training_history(
    results: Mapping[str, TrainingResult | dict[str, list[float | int]]],
    path: str | Path | None = None,
    show: bool = False,
):
    """Plot clean and Gaussian-averaged PINN losses when they are available."""

    plt = _pyplot(show)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = (
        ("Total loss", "clean_loss", "regularized_loss"),
        ("PDE loss", "clean_pde_loss", "gaussian_pde_loss"),
        ("Boundary loss", "clean_boundary_loss", "gaussian_boundary_loss"),
    )
    for label, result in results.items():
        history = result.history if isinstance(result, TrainingResult) else result
        steps = np.asarray(history.get("step", []))
        for axis, (title, clean_key, gaussian_key) in zip(axes, panels):
            for key, suffix, style in (
                (clean_key, "clean", "-"),
                (gaussian_key, "Gaussian mean", "--"),
            ):
                values = np.asarray(history.get(key, []), dtype=float)
                if not values.size or not np.isfinite(values).any():
                    continue
                count = min(steps.size, values.size)
                axis.plot(
                    steps[:count],
                    values[:count],
                    style,
                    label=f"{label} — {suffix}",
                    alpha=0.9,
                )
            axis.set(xlabel="Step", ylabel=title, title=title)
            axis.grid(alpha=0.25)
    for axis in axes:
        finite_positive = [
            value
            for line in axis.lines
            for value in np.asarray(line.get_ydata(), dtype=float)
            if np.isfinite(value) and value > 0.0
        ]
        if finite_positive:
            axis.set_yscale("log")
        if axis.lines:
            axis.legend(fontsize="small")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_poisson_solutions(
    models: Mapping[str, nn.Module],
    points,
    path: str | Path | None = None,
    show: bool = False,
):
    """Plot fields with one color scale shared by all solutions and predictions."""

    from .pinn import exact_poisson_solution

    if not models:
        raise ValueError("At least one model is required")
    plt = _pyplot(show)
    rows = len(models)
    fig, axes = plt.subplots(rows, 3, figsize=(11, 3.4 * rows), squeeze=False)
    resolution = int(points.evaluation_resolution)
    evaluated_rows = []
    solution_vmin = float("inf")
    solution_vmax = float("-inf")
    for label, model in models.items():
        parameter = next(model.parameters())
        coordinates = points.evaluation.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        was_training = model.training
        model.eval()
        with torch.no_grad():
            prediction = model(coordinates)
            if prediction.ndim == 2 and prediction.shape[1] == 1:
                prediction = prediction[:, 0]
            exact = exact_poisson_solution(coordinates)
            error = (prediction - exact).abs()
        model.train(was_training)
        exact = exact.detach().cpu()
        prediction = prediction.detach().cpu()
        error = error.detach().cpu()
        solution_vmin = min(
            solution_vmin,
            float(exact.min()),
            float(prediction.min()),
        )
        solution_vmax = max(
            solution_vmax,
            float(exact.max()),
            float(prediction.max()),
        )
        evaluated_rows.append((label, exact, prediction, error))

    for row, (label, exact, prediction, error) in enumerate(evaluated_rows):
        fields = (
            (exact, "Exact solution"),
            (prediction, f"{label} prediction"),
            (error, f"{label} absolute error"),
        )
        for column, (axis, (field, title)) in enumerate(zip(axes[row], fields)):
            is_solution = column < 2
            image = axis.imshow(
                field.reshape(resolution, resolution).T.numpy(),
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                aspect="equal",
                cmap="viridis" if is_solution else "magma",
                vmin=solution_vmin if is_solution else None,
                vmax=solution_vmax if is_solution else None,
            )
            axis.set(xlabel="x", ylabel="y", title=title)
            fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_space_time_pinn_solutions(
    models: Mapping[str, nn.Module],
    points,
    feature_map: Callable[[torch.Tensor], torch.Tensor],
    exact_solution: Callable[[torch.Tensor], torch.Tensor],
    path: str | Path | None = None,
    show: bool = False,
):
    """Plot fields with one color scale shared by all solutions and predictions."""

    if not models:
        raise ValueError("At least one model is required")
    plt = _pyplot(show)
    rows = len(models)
    fig, axes = plt.subplots(rows, 3, figsize=(11, 3.4 * rows), squeeze=False)
    space_resolution = int(points.space_resolution)
    time_resolution = int(points.time_resolution)
    evaluated_rows = []
    solution_vmin = float("inf")
    solution_vmax = float("-inf")
    for label, model in models.items():
        parameter = next(model.parameters())
        coordinates = points.evaluation.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        was_training = model.training
        model.eval()
        with torch.no_grad():
            prediction = model(feature_map(coordinates))
            if prediction.ndim == 2 and prediction.shape[1] == 1:
                prediction = prediction[:, 0]
            exact = exact_solution(coordinates)
            error = (prediction - exact).abs()
        model.train(was_training)
        exact = exact.detach().cpu()
        prediction = prediction.detach().cpu()
        error = error.detach().cpu()
        solution_vmin = min(
            solution_vmin,
            float(exact.min()),
            float(prediction.min()),
        )
        solution_vmax = max(
            solution_vmax,
            float(exact.max()),
            float(prediction.max()),
        )
        extent = (
            float(coordinates[:, 0].min()),
            float(coordinates[:, 0].max()),
            float(coordinates[:, 1].min()),
            float(coordinates[:, 1].max()),
        )
        evaluated_rows.append((label, exact, prediction, error, extent))

    for row, (label, exact, prediction, error, extent) in enumerate(evaluated_rows):
        fields = (
            (exact, "Exact solution"),
            (prediction, f"{label} prediction"),
            (error, f"{label} absolute error"),
        )
        for column, (axis, (field, title)) in enumerate(zip(axes[row], fields)):
            is_solution = column < 2
            image = axis.imshow(
                field.reshape(space_resolution, time_resolution).T.numpy(),
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis" if is_solution else "magma",
                vmin=solution_vmin if is_solution else None,
                vmax=solution_vmax if is_solution else None,
            )
            axis.set(xlabel="x", ylabel="t", title=title)
            fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_1d_pinn_solutions(
    models: Mapping[str, nn.Module],
    points,
    feature_map: Callable[[torch.Tensor], torch.Tensor],
    exact_solution: Callable[[torch.Tensor], torch.Tensor],
    path: str | Path | None = None,
    show: bool = False,
):
    """Plot exact, predicted, and absolute-error curves for a one-dimensional PINN."""

    if not models:
        raise ValueError("At least one model is required")
    plt = _pyplot(show)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    first_model = next(iter(models.values()))
    parameter = next(first_model.parameters())
    coordinates = points.evaluation.to(device=parameter.device, dtype=parameter.dtype)
    exact = exact_solution(coordinates).detach().cpu()
    x = coordinates[:, 0].detach().cpu()
    axes[0].plot(x, exact, color="black", linewidth=2.0, label="exact")

    for label, model in models.items():
        parameter = next(model.parameters())
        model_coordinates = points.evaluation.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        was_training = model.training
        model.eval()
        with torch.no_grad():
            prediction = model(feature_map(model_coordinates))
            if prediction.ndim == 2 and prediction.shape[1] == 1:
                prediction = prediction[:, 0]
            model_exact = exact_solution(model_coordinates)
            error = (prediction - model_exact).abs()
        model.train(was_training)
        axes[0].plot(x, prediction.detach().cpu(), label=label)
        axes[1].plot(x, error.detach().cpu(), label=label)

    axes[0].set(xlabel="x", ylabel="u(x)", title="Exact and predicted solution")
    axes[1].set(xlabel="x", ylabel="Absolute error", title="Pointwise error")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    return _finish(fig, path, show)
