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


def _finite_series(steps, values) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned, finite one-dimensional history values."""

    y_values = np.asarray(values, dtype=float)
    if y_values.ndim != 1 or not y_values.size:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x_values = np.asarray(steps, dtype=float)
    if not x_values.size:
        x_values = np.arange(y_values.size, dtype=float)
    count = min(x_values.size, y_values.size)
    x_values = x_values[:count]
    y_values = y_values[:count]
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[finite], y_values[finite]


def _smoothing_window(length: int, requested: int | None) -> int:
    """Choose an odd rolling-median window suited to the history length."""

    if requested is not None and requested < 1:
        raise ValueError("smoothing_window must be positive or None")
    if length < 9 and requested is None:
        return 1
    window = (
        requested
        if requested is not None
        else min(15, max(5, round(length * 0.02)))
    )
    window = min(int(window), length)
    if window % 2 == 0:
        window = window - 1 if window == length else window + 1
    return max(1, window)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    radius = window // 2
    return np.asarray(
        [
            np.median(values[max(0, index - radius) : index + radius + 1])
            for index in range(values.size)
        ],
        dtype=float,
    )


def _plot_history_series(
    axis,
    steps,
    values,
    *,
    label: str,
    smoothing_window: int | None,
    minimum_step: int | None = None,
    color=None,
    linestyle="-",
    marker: str = "o",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Plot faint observations and a readable rolling-median trend."""

    x_values, y_values = _finite_series(steps, values)
    if not y_values.size:
        return None
    if minimum_step is not None:
        visible = x_values >= minimum_step
        # Keep short runs useful rather than returning an empty chart.
        if visible.any():
            x_values = x_values[visible]
            y_values = y_values[visible]
    window = _smoothing_window(y_values.size, smoothing_window)
    trend = _rolling_median(y_values, window)
    if window > 1:
        raw_line = axis.plot(
            x_values,
            y_values,
            color=color,
            linestyle=linestyle,
            linewidth=0.65,
            alpha=0.14,
        )[0]
        color = raw_line.get_color()
        axis.plot(
            x_values,
            trend,
            color=color,
            linestyle=linestyle,
            linewidth=2.25,
            marker=marker,
            markevery=[0, len(x_values) - 1],
            markersize=4.5,
            label=label,
        )
    else:
        axis.plot(
            x_values,
            trend,
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            marker=marker,
            markevery=[0, len(x_values) - 1],
            markersize=4.5,
            alpha=0.9,
            label=label,
        )
    return x_values, y_values, trend, window


def _set_readable_y_scale(axis, series, *, prefer_log: bool = False) -> str:
    finite_parts = []
    for values in series:
        array = np.asarray(values, dtype=float).reshape(-1)
        finite_parts.append(array[np.isfinite(array)])
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return "linear"
    values = np.concatenate(finite_parts)
    positive = values[values > 0.0]
    dynamic_range = (
        float(positive.max() / positive.min()) if positive.size else 1.0
    )
    if positive.size == values.size and (prefer_log or dynamic_range >= 100.0):
        axis.set_yscale("log")
    elif (
        prefer_log
        and positive.size
        and np.all(values >= 0.0)
        and float(values.max()) > 0.0
    ):
        threshold = max(float(positive.min()), float(values.max()) * 1e-8)
        axis.set_yscale("symlog", linthresh=threshold)
    elif float(np.max(np.abs(values))) >= 10_000.0:
        from matplotlib.ticker import EngFormatter

        axis.yaxis.set_major_formatter(EngFormatter(sep=""))
    return axis.get_yscale()


def _metric_text(trend: np.ndarray, window: int) -> str:
    start = float(trend[0])
    final = float(trend[-1])
    best = float(np.min(trend))
    qualifier = f"{window}-step median" if window > 1 else "observed"
    if start > 0.0 and final > 0.0:
        ratio = start / final
        if ratio >= 1.0:
            change = f"Improvement  {ratio:.3g}x"
        else:
            change = f"Increase     {1.0 / ratio:.3g}x"
    else:
        change = f"Change       {final - start:+.3g}"
    return (
        f"{qualifier}\n"
        f"Start          {start:.3g}\n"
        f"Final          {final:.3g}\n"
        f"Best           {best:.3g}\n"
        f"{change}"
    )


def _annotate_metrics(axis, text: str) -> None:
    axis.text(
        0.98,
        0.97,
        text,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=8.5,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": axis.get_facecolor(),
            "edgecolor": axis.spines["top"].get_edgecolor(),
            "alpha": 0.88,
            "linewidth": 0.6,
        },
    )


def _style_history_axis(axis, ylabel: str) -> None:
    axis.set_xlabel("Training step")
    axis.set_ylabel(ylabel)
    axis.grid(which="major", alpha=0.25)
    axis.grid(which="minor", alpha=0.08)
    axis.margins(x=0.01)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_training_history(
    result: TrainingResult | dict[str, list[float | int]],
    path: str | Path | None = None,
    show: bool = False,
    *,
    smoothing_window: int | None = None,
    loss_start_step: int | None = 50,
):
    """Plot readable loss, balancedness, and schedule diagnostics.

    Per-step loss and balance measurements are shown faintly beneath a rolling
    median. By default its odd window is five percent of the history length,
    bounded between 5 and 15 samples; histories shorter than nine samples are
    left unsmoothed. The loss panel starts at step 50 by default so initial
    transients do not dominate its range; short runs remain untrimmed. Pass
    ``loss_start_step=None`` to show the complete loss history.
    """

    if loss_start_step is not None and loss_start_step < 0:
        raise ValueError("loss_start_step must be non-negative or None")
    plt = _pyplot(show)
    history = result.history if isinstance(result, TrainingResult) else result
    loss_length = max(
        (len(history.get(key, [])) for key in ("loss", "clean_loss")),
        default=0,
    )
    steps = history.get("step", list(range(loss_length)))

    available_panels = ["loss"]
    for key in ("layer_balance", "learning_rate", "sharpness_scale"):
        try:
            values = np.asarray(history.get(key, []), dtype=float)
        except (TypeError, ValueError):
            values = np.asarray([], dtype=float)
        if values.size and np.isfinite(values).any():
            available_panels.append(key)

    columns = 1 if len(available_panels) == 1 else 2
    rows = (len(available_panels) + columns - 1) // columns
    fig, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(11.5, 4.25 * rows),
        squeeze=False,
    )
    axes = axes_grid.reshape(-1)
    fig.suptitle("Training diagnostics", fontsize=14)

    for axis, panel in zip(axes, available_panels):
        if panel == "loss":
            candidates = (
                ("clean_loss", "Clean objective", "-", "tab:blue", "o"),
                (
                    "regularized_loss",
                    "Regularized objective",
                    (0, (5, 2)),
                    "tab:orange",
                    "s",
                ),
            )
            if not np.isfinite(
                np.asarray(history.get("clean_loss", []), dtype=float)
            ).any():
                candidates = (("loss", "Objective", "-", "tab:blue", "o"),)
            plotted = []
            for key, label, linestyle, color, marker in candidates:
                result_values = _plot_history_series(
                    axis,
                    steps,
                    history.get(key, []),
                    label=label,
                    smoothing_window=smoothing_window,
                    minimum_step=loss_start_step,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                )
                if result_values is not None:
                    plotted.append(result_values)
            gap_summary = ""
            if len(plotted) >= 2 and np.array_equal(
                plotted[0][0], plotted[1][0]
            ):
                x_values = plotted[0][0]
                clean_trend = plotted[0][2]
                regularized_trend = plotted[1][2]
                axis.fill_between(
                    x_values,
                    clean_trend,
                    regularized_trend,
                    color="tab:orange",
                    alpha=0.14,
                    linewidth=0.0,
                    label="Objective gap",
                )
                final_gap = float(regularized_trend[-1] - clean_trend[-1])
                if clean_trend[-1] != 0.0:
                    relative_gap = 100.0 * final_gap / abs(float(clean_trend[-1]))
                    gap_summary = (
                        f"\nFinal reg. gap {final_gap:.3g} ({relative_gap:+.3g}%)"
                    )
                else:
                    gap_summary = f"\nFinal reg. gap {final_gap:.3g}"
            scale = _set_readable_y_scale(
                axis,
                [values for _, values, _, _ in plotted],
                prefer_log=True,
            )
            loss_title = "Objective loss"
            if plotted and loss_start_step is not None:
                first_visible_step = float(plotted[0][0][0])
                if first_visible_step >= loss_start_step:
                    loss_title += f" (from step {first_visible_step:g})"
            axis.set_title(loss_title, loc="left")
            _style_history_axis(
                axis,
                "Loss" if scale == "linear" else f"Loss ({scale} scale)",
            )
            if plotted:
                _, _, trend, window = plotted[0]
                _annotate_metrics(axis, _metric_text(trend, window) + gap_summary)
                axis.legend(
                    loc="lower left",
                    frameon=False,
                    title=(
                        f"Bold = {window}-step rolling median"
                        if window > 1
                        else None
                    ),
                )

        elif panel == "layer_balance":
            balance = np.asarray(history[panel], dtype=float)
            if balance.ndim == 1:
                balance = balance[:, None]
            plotted = []
            for index in range(balance.shape[1]):
                values = _plot_history_series(
                    axis,
                    steps,
                    balance[:, index],
                    label=f"Layers {index + 1} and {index + 2}",
                    smoothing_window=smoothing_window,
                )
                if values is not None:
                    plotted.append(values)
            scale = _set_readable_y_scale(
                axis,
                [values for _, values, _, _ in plotted],
                prefer_log=True,
            )
            axis.set_title("Layer imbalance", loc="left")
            _style_history_axis(
                axis,
                "Gram discrepancy"
                if scale == "linear"
                else f"Gram discrepancy ({scale} scale)",
            )
            if plotted:
                window = plotted[0][3]
                axis.legend(
                    frameon=False,
                    fontsize="small",
                    ncols=2 if len(plotted) > 2 else 1,
                    title=(
                        f"Bold = {window}-step rolling median"
                        if window > 1
                        else None
                    ),
                )

        else:
            label = "Learning rate" if panel == "learning_rate" else "Sharpness scale"
            x_values, y_values = _finite_series(steps, history[panel])
            line = axis.plot(x_values, y_values, linewidth=2.0)[0]
            if y_values.size:
                axis.scatter(
                    x_values[-1], y_values[-1], color=line.get_color(), s=20, zorder=3
                )
                summary = (
                    f"Start  {y_values[0]:.3g}\n"
                    f"Final  {y_values[-1]:.3g}\n"
                    f"Range  {y_values.min():.3g} to {y_values.max():.3g}"
                )
                _annotate_metrics(axis, summary)
            scale = _set_readable_y_scale(axis, [y_values])
            axis.set_title(label, loc="left")
            _style_history_axis(
                axis,
                label if scale == "linear" else f"{label} ({scale} scale)",
            )

    for axis in axes[len(available_panels) :]:
        axis.remove()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
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

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.suptitle(
        f"Sharpness interpolation (scale = {result.sharpness_scale:.8g})"
    )
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
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return _finish(fig, path, show)


def plot_pinn_training_history(
    results: Mapping[str, TrainingResult | dict[str, list[float | int]]],
    path: str | Path | None = None,
    show: bool = False,
    *,
    smoothing_window: int | None = None,
    loss_start_step: int | None = 50,
):
    """Plot PINN losses after an optional warm-up on readable scales."""

    if loss_start_step is not None and loss_start_step < 0:
        raise ValueError("loss_start_step must be non-negative or None")
    plt = _pyplot(show)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.suptitle("PINN training diagnostics", fontsize=14)
    panels = (
        ("Total loss", "clean_loss", "regularized_loss"),
        ("PDE loss", "clean_pde_loss", "gaussian_pde_loss"),
        ("Boundary loss", "clean_boundary_loss", "gaussian_boundary_loss"),
    )
    panel_values: list[list[np.ndarray]] = [[], [], []]
    panel_windows: list[list[int]] = [[], [], []]
    panel_starts: list[list[float]] = [[], [], []]
    for result_index, (label, result) in enumerate(results.items()):
        history = result.history if isinstance(result, TrainingResult) else result
        steps = np.asarray(history.get("step", []))
        for panel_index, (axis, (title, clean_key, gaussian_key)) in enumerate(
            zip(axes, panels)
        ):
            for key, suffix, style in (
                (clean_key, "clean", "-"),
                (gaussian_key, "Gaussian mean", "--"),
            ):
                plotted = _plot_history_series(
                    axis,
                    steps,
                    history.get(key, []),
                    label=f"{label} · {suffix}",
                    smoothing_window=smoothing_window,
                    minimum_step=loss_start_step,
                    color=f"C{result_index % 10}",
                    linestyle=style,
                )
                if plotted is not None:
                    x_values, values, _, window = plotted
                    panel_values[panel_index].append(values)
                    panel_windows[panel_index].append(window)
                    panel_starts[panel_index].append(float(x_values[0]))

    for panel_index, (axis, (title, _, _)) in enumerate(zip(axes, panels)):
        scale = _set_readable_y_scale(
            axis,
            panel_values[panel_index],
            prefer_log=True,
        )
        panel_title = title
        if panel_starts[panel_index] and loss_start_step is not None:
            first_visible_step = min(panel_starts[panel_index])
            if first_visible_step >= loss_start_step:
                panel_title += f" (from step {first_visible_step:g})"
        axis.set_title(panel_title, loc="left")
        _style_history_axis(
            axis,
            title if scale == "linear" else f"{title} ({scale} scale)",
        )
        if axis.lines:
            smoothed = [window for window in panel_windows[panel_index] if window > 1]
            legend_title = (
                f"Bold = {max(smoothed)}-step rolling median"
                if smoothed
                else None
            )
            axis.legend(fontsize="small", frameon=False, title=legend_title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
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
