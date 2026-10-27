"""Grid-search gradient optimizers for configurable 2D Poisson PINNs.

The default problem is a manufactured solution on the unit square,

    -Delta u = f  in Omega,       u = g  on boundary(Omega),

with exact harmonic boundary choices ``zero``, ``constant``, ``affine``,
``bilinear``, and ``harmonic_polynomial``. Neither the right-hand side,
boundary values, nor model are hard-coded in
the search loop.  To use a different problem or model, point ``problem_factory``
or ``model_factory`` at an importable ``"module:function"``.  A problem factory
receives one setup dictionary and returns a :class:`PoissonProblem`; a model
factory receives ``(model_config, seed)`` and returns ``torch.nn.Module``.

Examples (run from the repository root)::

    python examples/grid_search_poisson_sgd.py --smoke-test
    python examples/grid_search_poisson_sgd.py --write-default-config search.json
    python examples/grid_search_poisson_sgd.py --config search.json

The JSON config's ``grid`` is a Cartesian product of dotted config paths.  It
can therefore tune optimizer settings as well as loss weights and architecture,
for example ``training.learning_rate``, ``training.batch_size``,
``training.optimizer.momentum``, ``loss_weights.boundary``, or ``model.width``.
The main output is ``optimal_optimizer_config.json`` (with the former
``optimal_sgd_config.json`` retained as a compatibility alias); complete trial
data are retained in ``grid_search_results.json`` and ``trials.csv``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import itertools
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ssam import (
    build_model,
    build_optimizer,
    poisson_boundary_values,
    resolve_poisson_boundary_condition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PoissonProblem:
    """Callbacks and rectangular domain defining a scalar 2D Poisson problem."""

    name: str
    forcing: Callable[[torch.Tensor], torch.Tensor]
    boundary_value: Callable[[torch.Tensor], torch.Tensor]
    exact_solution: Callable[[torch.Tensor], torch.Tensor] | None = None
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0),
        (0.0, 1.0),
    )


@dataclass(frozen=True)
class PointSets:
    train_interior: torch.Tensor
    train_boundary: torch.Tensor
    validation_interior: torch.Tensor
    validation_boundary: torch.Tensor
    evaluation: torch.Tensor


DEFAULT_GRID_SEARCH_CONFIG: dict[str, Any] = {
    "output_dir": "outputs/poisson_sgd_grid_search",
    # Set either factory to e.g. "my_experiment:make_problem".  Null selects
    # the built-in manufactured-solution problem/model builders below.
    "problem_factory": None,
    "model_factory": None,
    "setups": [
        {
            "name": "sine_1x1_zero_boundary",
            "problem": {
                "amplitude": 1.0,
                "modes": [1, 1],
                "boundary_condition": {"name": "zero"},
                "bounds": [[0.0, 1.0], [0.0, 1.0]],
            },
            "data": {
                "train_interior": 512,
                "train_boundary": 256,
                "validation_interior": 1024,
                "validation_boundary": 512,
                "evaluation_resolution": 51,
                "dtype": "float32",
                "seed": 31415,
            },
        },
        {
            "name": "sine_2x1_nonzero_boundary",
            "problem": {
                "amplitude": 0.75,
                "modes": [2, 1],
                "boundary_condition": {
                    "name": "bilinear",
                    "constant": 0.1,
                    "x": 0.2,
                    "y": -0.1,
                    "xy": 0.15,
                },
                "bounds": [[0.0, 1.0], [0.0, 1.0]],
            },
            "data": {
                "train_interior": 512,
                "train_boundary": 256,
                "validation_interior": 1024,
                "validation_boundary": 512,
                "evaluation_resolution": 51,
                "dtype": "float32",
                "seed": 27182,
            },
        },
    ],
    "model": {
        "name": "poisson_grid_search",
        "type": "mlp",
        "input_dim": 2,
        "output_dim": 1,
        "depth": 3,
        "width": 48,
        "activation": "tanh",
        "output_activation": "identity",
        "bias": True,
        "parameter_init": {"type": "xavier_uniform", "seed": 0},
    },
    "training": {
        "algorithm": "sgd",
        "steps": 1600,
        "batch_size": 64,
        # A number means constant LR.  Schedule dictionaries are also accepted;
        # see learning_rate_at() for constant, step, exponential, and cosine.
        "learning_rate": 1e-3,
        "optimizer": {
            "name": "sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "dampening": 0.0,
            "nesterov": False,
        },
        "max_grad_norm": None,
        "device": "auto",
    },
    "loss_weights": {"pde": 1.0, "boundary": 7.5},
    "grid": {
        "training.learning_rate": [1e-3, 1e-2],
        "training.batch_size": [64, 32],
        # Complete optimizer dictionaries keep optimizer-specific options from
        # leaking into other choices (for example momentum into Adam).
        "training.optimizer": [
            {
                "name": "sgd",
                "momentum": 0.0,
                "weight_decay": 0.0,
                "dampening": 0.0,
                "nesterov": False,
            },
            {
                "name": "sgd",
                "momentum": 0.9,
                "weight_decay": 0.0,
                "dampening": 0.0,
                "nesterov": False,
            },
            {
                "name": "sgd",
                "momentum": 2.0,
                "weight_decay": 0.0,
                "dampening": 0.0,
                "nesterov": False,
            },
            {
                "name": "adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
            {
                "name": "adagrad",
                "lr_decay": 0.0,
                "initial_accumulator_value": 0.0,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
        ],
    },
    # Repeats share seeds across candidates, so candidate comparisons use the
    # same initialization and minibatch permutations.
    "repeat_seeds": [1000,2000],
    "selection": {
        # Always available: validation_loss, pde_residual_rmse, boundary_rmse.
        # relative_l2_error is available when exact_solution is supplied.
        "metric": "validation_loss",
        # Fixed evaluation weights prevent candidates that tune loss_weights
        # from winning merely by making their own training weights smaller.
        "pde_weight": 1.0,
        "boundary_weight": 1.0,
    },
    "evaluation_batch_size": 1024,
}


def manufactured_sine_problem(setup: Mapping[str, Any]) -> PoissonProblem:
    """Build the default manufactured Poisson problem from one setup."""

    options = setup.get("problem", {})
    amplitude = float(options.get("amplitude", 1.0))
    modes = tuple(int(value) for value in options.get("modes", (1, 1)))
    if "boundary_condition" in options:
        boundary_condition = resolve_poisson_boundary_condition(
            options["boundary_condition"]
        )
    else:
        # Backward-compatible shorthand used by early search configs.
        affine = tuple(
            float(value)
            for value in options.get("affine", (0.0, 0.0, 0.0))
        )
        if len(affine) != 3:
            raise ValueError(
                "problem.affine must be [constant, x_coefficient, y_coefficient]"
            )
        boundary_condition = resolve_poisson_boundary_condition(
            {
                "name": "affine",
                "constant": affine[0],
                "x": affine[1],
                "y": affine[2],
            }
        )
    raw_bounds = options.get("bounds", ((0.0, 1.0), (0.0, 1.0)))
    bounds = tuple(tuple(float(value) for value in axis) for axis in raw_bounds)
    if len(modes) != 2 or any(value < 1 for value in modes):
        raise ValueError("problem.modes must contain two positive integers")
    _validate_bounds(bounds)
    (x0, x1), (y0, y1) = bounds
    wave_x = modes[0] * math.pi / (x1 - x0)
    wave_y = modes[1] * math.pi / (y1 - y0)

    def sine(points: torch.Tensor) -> torch.Tensor:
        return amplitude * torch.sin(wave_x * (points[:, 0] - x0)) * torch.sin(
            wave_y * (points[:, 1] - y0)
        )

    def exact_solution(points: torch.Tensor) -> torch.Tensor:
        return sine(points) + poisson_boundary_values(
            points,
            boundary_condition,
        )

    def forcing(points: torch.Tensor) -> torch.Tensor:
        return (wave_x**2 + wave_y**2) * sine(points)

    return PoissonProblem(
        name=str(setup.get("name", "manufactured_sine")),
        forcing=forcing,
        boundary_value=exact_solution,
        exact_solution=exact_solution,
        bounds=bounds,  # type: ignore[arg-type]
    )


def default_model_factory(model_config: Mapping[str, Any], seed: int) -> nn.Module:
    """Build one repository MLP, overriding only its initialization seed."""

    resolved = copy.deepcopy(dict(model_config))
    initialization = resolved.setdefault("parameter_init", {"type": "xavier_uniform"})
    if isinstance(initialization, str):
        initialization = {"type": initialization}
        resolved["parameter_init"] = initialization
    initialization["seed"] = int(seed)
    return build_model({"model": resolved})


def _validate_bounds(bounds: Sequence[Sequence[float]]) -> None:
    if len(bounds) != 2 or any(len(axis) != 2 for axis in bounds):
        raise ValueError("bounds must be [[x_min, x_max], [y_min, y_max]]")
    if any(not math.isfinite(value) for axis in bounds for value in axis):
        raise ValueError("bounds must be finite")
    if any(axis[0] >= axis[1] for axis in bounds):
        raise ValueError("Each lower domain bound must be less than its upper bound")


def _load_factory(specification: str | None, fallback: Callable[..., Any]) -> Callable[..., Any]:
    if not specification:
        return fallback
    if ":" not in specification:
        raise ValueError("Factory paths must have the form 'module:function'")
    module_name, attribute_name = specification.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError(f"Configured factory {specification!r} is not callable")
    return factory


def _coerce_problem(value: Any, setup_name: str) -> PoissonProblem:
    if isinstance(value, PoissonProblem):
        problem = value
    elif isinstance(value, Mapping):
        problem = PoissonProblem(
            name=str(value.get("name", setup_name)),
            forcing=value["forcing"],
            boundary_value=value["boundary_value"],
            exact_solution=value.get("exact_solution"),
            bounds=tuple(tuple(axis) for axis in value.get("bounds", ((0.0, 1.0), (0.0, 1.0)))),  # type: ignore[arg-type]
        )
    else:
        required = ("forcing", "boundary_value", "bounds")
        if not all(hasattr(value, name) for name in required):
            raise TypeError("A problem factory must return PoissonProblem or a compatible mapping/object")
        problem = PoissonProblem(
            name=str(getattr(value, "name", setup_name)),
            forcing=value.forcing,
            boundary_value=value.boundary_value,
            exact_solution=getattr(value, "exact_solution", None),
            bounds=tuple(tuple(axis) for axis in value.bounds),  # type: ignore[arg-type]
        )
    if not callable(problem.forcing) or not callable(problem.boundary_value):
        raise TypeError("Problem forcing and boundary_value must be callable")
    if problem.exact_solution is not None and not callable(problem.exact_solution):
        raise TypeError("Problem exact_solution must be callable or None")
    _validate_bounds(problem.bounds)
    return problem


def _dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name.lower()]
    except KeyError as exc:
        raise ValueError("data.dtype must be 'float32' or 'float64'") from exc


def _uniform_points(
    count: int,
    bounds: Sequence[Sequence[float]],
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    points = torch.rand((count, 2), generator=generator, dtype=dtype)
    for dimension, (lower, upper) in enumerate(bounds):
        points[:, dimension].mul_(upper - lower).add_(lower)
    return points


def _boundary_points(
    count: int,
    bounds: Sequence[Sequence[float]],
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    points = _uniform_points(count, bounds, generator, dtype)
    (x0, x1), (y0, y1) = bounds
    sides = torch.arange(count) % 4
    points[sides == 0, 0] = x0
    points[sides == 1, 0] = x1
    points[sides == 2, 1] = y0
    points[sides == 3, 1] = y1
    return points


def build_point_sets(problem: PoissonProblem, data: Mapping[str, Any]) -> PointSets:
    """Create deterministic, disjoint training and validation collocation sets."""

    counts = {
        "train_interior": int(data.get("train_interior", 512)),
        "train_boundary": int(data.get("train_boundary", 256)),
        "validation_interior": int(data.get("validation_interior", 1024)),
        "validation_boundary": int(data.get("validation_boundary", 512)),
    }
    if any(value < 1 for value in counts.values()):
        raise ValueError("All train/validation point counts must be positive")
    resolution = int(data.get("evaluation_resolution", 51))
    if resolution < 2:
        raise ValueError("evaluation_resolution must be at least 2")
    dtype = _dtype(str(data.get("dtype", "float32")))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(data.get("seed", 0)))
    train_interior = _uniform_points(
        counts["train_interior"], problem.bounds, generator, dtype
    )
    train_boundary = _boundary_points(
        counts["train_boundary"], problem.bounds, generator, dtype
    )
    validation_interior = _uniform_points(
        counts["validation_interior"], problem.bounds, generator, dtype
    )
    validation_boundary = _boundary_points(
        counts["validation_boundary"], problem.bounds, generator, dtype
    )
    axes = [
        torch.linspace(lower, upper, resolution, dtype=dtype)
        for lower, upper in problem.bounds
    ]
    grid_x, grid_y = torch.meshgrid(*axes, indexing="ij")
    evaluation = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    return PointSets(
        train_interior,
        train_boundary,
        validation_interior,
        validation_boundary,
        evaluation,
    )


def _scalar_prediction(model: nn.Module, points: torch.Tensor) -> torch.Tensor:
    prediction = model(points)
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if prediction.ndim != 1 or prediction.shape[0] != points.shape[0]:
        raise ValueError("The model must return shape (N,) or (N, 1)")
    return prediction


def _scalar_callback_values(
    values: torch.Tensor,
    point_count: int,
    callback_name: str,
) -> torch.Tensor:
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.shape[0] != point_count:
        raise ValueError(f"{callback_name}(points) must return shape (N,) or (N, 1)")
    return values


def poisson_residual(
    model: nn.Module,
    coordinates: torch.Tensor,
    forcing: Callable[[torch.Tensor], torch.Tensor],
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Compute ``-Delta u_theta - f`` for arbitrary forcing callbacks."""

    with torch.enable_grad():
        points = coordinates.detach().clone().requires_grad_(True)
        prediction = _scalar_prediction(model, points)
        gradient = torch.autograd.grad(
            prediction,
            points,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        laplacian = torch.zeros_like(prediction)
        if gradient is not None:
            for dimension in range(2):
                second_derivatives = torch.autograd.grad(
                    gradient[:, dimension],
                    points,
                    grad_outputs=torch.ones_like(gradient[:, dimension]),
                    create_graph=create_graph,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                # Linear models (and models independent of one coordinate)
                # have an exact zero second derivative.  Autograd represents
                # that case as an unused input instead of materializing zeros.
                second = (
                    torch.zeros_like(prediction)
                    if second_derivatives is None
                    else second_derivatives[:, dimension]
                )
                laplacian = laplacian + second
        expected = _scalar_callback_values(forcing(points), len(points), "forcing")
        return -laplacian - expected


def loss_components(
    model: nn.Module,
    interior: torch.Tensor,
    boundary: torch.Tensor,
    problem: PoissonProblem,
    weights: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pde_weight = float(weights["pde"])
    boundary_weight = float(weights["boundary"])
    if pde_weight < 0.0 or boundary_weight < 0.0:
        raise ValueError("loss_weights must be non-negative")
    if pde_weight == 0.0 and boundary_weight == 0.0:
        raise ValueError("At least one loss weight must be positive")
    residual = poisson_residual(model, interior, problem.forcing, create_graph=True)
    pde_loss = residual.square().mean()
    prediction = _scalar_prediction(model, boundary)
    target = _scalar_callback_values(
        problem.boundary_value(boundary), len(boundary), "boundary_value"
    )
    boundary_loss = (prediction - target).square().mean()
    total = pde_weight * pde_loss + boundary_weight * boundary_loss
    return total, pde_loss, boundary_loss


class _PermutationBatches:
    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        self.size = size
        self.batch_size = min(size, max(1, batch_size))
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.permutation = torch.empty(0, dtype=torch.long)
        self.cursor = size

    def next(self, device: torch.device) -> torch.Tensor:
        if self.batch_size == self.size:
            return torch.arange(self.size, device=device)
        if self.cursor + self.batch_size > self.size:
            self.permutation = torch.randperm(self.size, generator=self.generator)
            self.cursor = 0
        result = self.permutation[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return result.to(device)


def _device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def learning_rate_at(specification: Any, step: int, total_steps: int) -> float:
    """Resolve a scalar or a small JSON-friendly learning-rate schedule."""

    if isinstance(specification, (int, float)):
        value = float(specification)
    elif isinstance(specification, Mapping):
        name = str(specification.get("name", "constant")).lower()
        initial = float(specification.get("initial", specification.get("value", 0.0)))
        if name == "constant":
            value = initial
        elif name == "step":
            step_size = int(specification.get("step_size", 100))
            if step_size < 1:
                raise ValueError("A step LR schedule needs positive step_size")
            value = initial * float(specification.get("gamma", 0.1)) ** (step // step_size)
        elif name == "exponential":
            value = initial * float(specification.get("gamma", 0.99)) ** step
        elif name == "cosine":
            minimum = float(specification.get("minimum", 0.0))
            progress = step / max(1, total_steps - 1)
            value = minimum + 0.5 * (initial - minimum) * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown learning-rate schedule {name!r}")
    else:
        raise TypeError("training.learning_rate must be a number or mapping")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Every learning rate must be positive and finite")
    return value


def train_once(
    model: nn.Module,
    points: PointSets,
    problem: PoissonProblem,
    candidate: Mapping[str, Any],
    seed: int,
) -> dict[str, float]:
    """Train one candidate/repeat with deterministic initialization and batches."""

    training = candidate["training"]
    if str(training.get("algorithm", "sgd")).lower() != "sgd":
        raise ValueError("This script only searches training.algorithm='sgd'")
    steps = int(training["steps"])
    if steps < 1:
        raise ValueError("training.steps must be positive")
    device = _device(str(training.get("device", "auto")))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model.to(device)
    parameter = next((value for value in model.parameters() if value.requires_grad), None)
    if parameter is None:
        raise ValueError("The model has no trainable parameters")
    dtype = parameter.dtype
    interior = points.train_interior.to(device=device, dtype=dtype)
    boundary = points.train_boundary.to(device=device, dtype=dtype)
    requested_batch = int(training["batch_size"])
    if requested_batch < 1:
        raise ValueError("training.batch_size must be positive")
    boundary_batch_size = max(
        1, math.ceil(min(1.0, requested_batch / len(interior)) * len(boundary))
    )
    interior_batches = _PermutationBatches(len(interior), requested_batch, seed + 1)
    boundary_batches = _PermutationBatches(len(boundary), boundary_batch_size, seed + 2)
    optimizer = build_optimizer(
        model.parameters(),
        training.get("optimizer", {}),
        learning_rate=learning_rate_at(training["learning_rate"], 0, steps),
    )
    max_grad_norm = training.get("max_grad_norm")
    max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
    model.train()
    final = {"training_loss": math.nan, "training_pde_loss": math.nan, "training_boundary_loss": math.nan}
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(training["learning_rate"], step, steps)
        interior_batch = interior.index_select(0, interior_batches.next(device))
        boundary_batch = boundary.index_select(0, boundary_batches.next(device))
        optimizer.zero_grad(set_to_none=True)
        total, pde_loss, boundary_loss = loss_components(
            model, interior_batch, boundary_batch, problem, candidate["loss_weights"]
        )
        if not torch.isfinite(total).item():
            raise FloatingPointError(f"Non-finite training loss at step {step}")
        total.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        final = {
            "training_loss": float(total.detach()),
            "training_pde_loss": float(pde_loss.detach()),
            "training_boundary_loss": float(boundary_loss.detach()),
        }
    return final


def _residual_rmse(
    model: nn.Module,
    points: torch.Tensor,
    problem: PoissonProblem,
    batch_size: int,
) -> float:
    squared_sum = 0.0
    count = 0
    for batch in points.split(batch_size):
        residual = poisson_residual(model, batch, problem.forcing, create_graph=False)
        squared_sum += float(residual.detach().square().sum())
        count += residual.numel()
    return math.sqrt(squared_sum / count)


def evaluate(
    model: nn.Module,
    points: PointSets,
    problem: PoissonProblem,
    selection: Mapping[str, Any],
    evaluation_batch_size: int,
) -> dict[str, float | None]:
    if evaluation_batch_size < 1:
        raise ValueError("evaluation_batch_size must be positive")
    selection_pde_weight = float(selection.get("pde_weight", 1.0))
    selection_boundary_weight = float(selection.get("boundary_weight", 1.0))
    if selection_pde_weight < 0.0 or selection_boundary_weight < 0.0:
        raise ValueError("Selection weights must be non-negative")
    if selection_pde_weight == 0.0 and selection_boundary_weight == 0.0:
        raise ValueError("At least one selection weight must be positive")
    parameter = next(value for value in model.parameters() if value.requires_grad)
    device, dtype = parameter.device, parameter.dtype
    interior = points.validation_interior.to(device=device, dtype=dtype)
    boundary = points.validation_boundary.to(device=device, dtype=dtype)
    evaluation_points = points.evaluation.to(device=device, dtype=dtype)
    was_training = model.training
    model.eval()
    try:
        pde_rmse = _residual_rmse(model, interior, problem, evaluation_batch_size)
        with torch.no_grad():
            boundary_prediction = _scalar_prediction(model, boundary)
            boundary_target = _scalar_callback_values(
                problem.boundary_value(boundary), len(boundary), "boundary_value"
            )
            boundary_rmse = float((boundary_prediction - boundary_target).square().mean().sqrt())
            relative_l2_error = None
            if problem.exact_solution is not None:
                prediction = _scalar_prediction(model, evaluation_points)
                exact = _scalar_callback_values(
                    problem.exact_solution(evaluation_points),
                    len(evaluation_points),
                    "exact_solution",
                )
                denominator = float(exact.square().sum().sqrt())
                if denominator > torch.finfo(dtype).tiny:
                    relative_l2_error = float((prediction - exact).square().sum().sqrt()) / denominator
        validation_loss = (
            selection_pde_weight * pde_rmse**2
            + selection_boundary_weight * boundary_rmse**2
        )
        metrics: dict[str, float | None] = {
            "validation_loss": validation_loss,
            "pde_residual_rmse": pde_rmse,
            "boundary_rmse": boundary_rmse,
            "relative_l2_error": relative_l2_error,
        }
        metric_name = str(selection.get("metric", "validation_loss"))
        if metric_name not in metrics:
            raise ValueError(f"Unknown selection.metric {metric_name!r}")
        if metrics[metric_name] is None:
            raise ValueError(
                f"selection.metric={metric_name!r} requires an exact_solution callback"
            )
        finite_values = [value for value in metrics.values() if value is not None]
        if not all(math.isfinite(value) for value in finite_values):
            raise FloatingPointError("Evaluation produced a non-finite metric")
        metrics["selection_score"] = metrics[metric_name]
        return metrics
    finally:
        model.train(was_training)


def _set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid grid path {path!r}")
    target: dict[str, Any] = config
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise KeyError(f"Grid path {path!r} does not address a config mapping")
        target = child
    if parts[-1] not in target:
        raise KeyError(f"Grid path {path!r} does not exist in the base config")
    target[parts[-1]] = copy.deepcopy(value)


def expand_grid(config: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return ``(chosen_hyperparameters, resolved_candidate)`` combinations."""

    grid = config.get("grid", {})
    if not isinstance(grid, Mapping):
        raise TypeError("grid must be a mapping of dotted paths to lists")
    paths = list(grid)
    values = []
    for path in paths:
        choices = grid[path]
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise ValueError(f"grid[{path!r}] must be a non-empty list")
        values.append(choices)
    base = {
        "model": copy.deepcopy(config["model"]),
        "training": copy.deepcopy(config["training"]),
        "loss_weights": copy.deepcopy(config["loss_weights"]),
    }
    combinations = itertools.product(*values) if paths else [()]
    expanded = []
    for combination in combinations:
        candidate = copy.deepcopy(base)
        chosen = dict(zip(paths, combination))
        for path, value in chosen.items():
            _set_dotted(candidate, path, value)
        expanded.append((chosen, candidate))
    return expanded


def _mean_standard_deviation(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _summarize_candidate(
    candidate_id: int,
    hyperparameters: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trials: list[dict[str, Any]],
    setup_names: Sequence[str],
    repeat_count: int,
) -> dict[str, Any]:
    setup_metrics: dict[str, Any] = {}
    complete = True
    for setup_name in setup_names:
        successful = [
            trial for trial in trials if trial["setup"] == setup_name and trial["status"] == "ok"
        ]
        if len(successful) != repeat_count:
            complete = False
        aggregate: dict[str, Any] = {
            "successful_repeats": len(successful),
            "required_repeats": repeat_count,
        }
        for metric in (
            "selection_score",
            "validation_loss",
            "pde_residual_rmse",
            "boundary_rmse",
            "relative_l2_error",
        ):
            values = [float(trial[metric]) for trial in successful if trial.get(metric) is not None]
            aggregate[metric] = _mean_standard_deviation(values) if values else None
        setup_metrics[setup_name] = aggregate
    score = None
    if complete:
        score = statistics.fmean(
            setup_metrics[name]["selection_score"]["mean"] for name in setup_names
        )
    return {
        "candidate_id": candidate_id,
        "status": "complete" if complete else "incomplete",
        "score_across_setups": score,
        "hyperparameters": copy.deepcopy(dict(hyperparameters)),
        "resolved_config": copy.deepcopy(dict(candidate)),
        "setup_metrics": setup_metrics,
        "trials": trials,
    }


def _winner_payload(candidate: Mapping[str, Any], score: float) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "score": score,
        "hyperparameters": candidate["hyperparameters"],
        "resolved_config": candidate["resolved_config"],
    }


def _write_csv(path: Path, candidates: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "setup",
        "repeat_seed",
        "status",
        "selection_score",
        "validation_loss",
        "pde_residual_rmse",
        "boundary_rmse",
        "relative_l2_error",
        "training_loss",
        "elapsed_seconds",
        "error",
        "hyperparameters",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            for trial in candidate["trials"]:
                row = {field: trial.get(field) for field in fields}
                row["candidate_id"] = candidate["candidate_id"]
                row["hyperparameters"] = json.dumps(candidate["hyperparameters"], sort_keys=True)
                writer.writerow(row)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_grid_search(config: Mapping[str, Any]) -> Path:
    """Run all candidates and write the selected optimizer configuration."""

    candidates = expand_grid(config)
    setups = list(config.get("setups", ()))
    seeds = [int(seed) for seed in config.get("repeat_seeds", ())]
    if not setups:
        raise ValueError("At least one setup is required")
    if not seeds:
        raise ValueError("At least one repeat seed is required")
    setup_names = [str(setup.get("name", f"setup_{index}")) for index, setup in enumerate(setups)]
    if len(set(setup_names)) != len(setup_names):
        raise ValueError("Setup names must be unique")
    problem_factory = _load_factory(config.get("problem_factory"), manufactured_sine_problem)
    model_factory = _load_factory(config.get("model_factory"), default_model_factory)
    prepared = []
    for setup, setup_name in zip(setups, setup_names):
        problem = _coerce_problem(problem_factory(setup), setup_name)
        prepared.append((setup_name, problem, build_point_sets(problem, setup.get("data", {}))))

    output_dir = Path(str(config.get("output_dir", "outputs/poisson_sgd_grid_search")))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    total_trials = len(candidates) * len(prepared) * len(seeds)
    print(f"Running {len(candidates)} candidates x {len(prepared)} setups x {len(seeds)} seeds = {total_trials} trials")

    summaries = []
    trial_number = 0
    for candidate_id, (hyperparameters, candidate) in enumerate(candidates):
        trials: list[dict[str, Any]] = []
        for setup_name, problem, points in prepared:
            for seed in seeds:
                trial_number += 1
                started = time.perf_counter()
                trial: dict[str, Any] = {
                    "setup": setup_name,
                    "repeat_seed": seed,
                    "status": "ok",
                    "error": None,
                }
                try:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    model = model_factory(copy.deepcopy(candidate["model"]), seed)
                    if not isinstance(model, nn.Module):
                        raise TypeError("The model factory must return torch.nn.Module")
                    trial.update(train_once(model, points, problem, candidate, seed))
                    trial.update(
                        evaluate(
                            model,
                            points,
                            problem,
                            config["selection"],
                            int(config.get("evaluation_batch_size", 1024)),
                        )
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    trial["status"] = "failed"
                    trial["error"] = f"{type(exc).__name__}: {exc}"
                trial["elapsed_seconds"] = time.perf_counter() - started
                trials.append(trial)
                score_text = (
                    f"score={trial['selection_score']:.5g}"
                    if trial["status"] == "ok"
                    else trial["error"]
                )
                print(f"[{trial_number:>4}/{total_trials}] candidate={candidate_id:>3} setup={setup_name} seed={seed}: {score_text}")
        summaries.append(
            _summarize_candidate(
                candidate_id, hyperparameters, candidate, trials, setup_names, len(seeds)
            )
        )

    eligible = [item for item in summaries if item["status"] == "complete"]
    if not eligible:
        results = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": copy.deepcopy(config["selection"]),
            "candidates": summaries,
        }
        _write_json(output_dir / "grid_search_results.json", results)
        _write_csv(output_dir / "trials.csv", summaries)
        raise RuntimeError(
            f"Every candidate had at least one failed trial; inspect {output_dir / 'grid_search_results.json'}"
        )
    overall = min(eligible, key=lambda item: item["score_across_setups"])
    best_per_setup = {}
    for setup_name in setup_names:
        winner = min(
            eligible,
            key=lambda item: item["setup_metrics"][setup_name]["selection_score"]["mean"],
        )
        best_per_setup[setup_name] = _winner_payload(
            winner,
            winner["setup_metrics"][setup_name]["selection_score"]["mean"],
        )
    optimal = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": copy.deepcopy(config["selection"]),
        "best_overall": _winner_payload(overall, overall["score_across_setups"]),
        "best_per_setup": best_per_setup,
    }
    results = {
        "created_at": optimal["created_at"],
        "selection": copy.deepcopy(config["selection"]),
        "problem_factory": config.get("problem_factory"),
        "model_factory": config.get("model_factory"),
        "setups": copy.deepcopy(setups),
        "repeat_seeds": seeds,
        "best_overall_candidate_id": overall["candidate_id"],
        "candidates": summaries,
    }
    _write_json(output_dir / "optimal_optimizer_config.json", optimal)
    # Preserve the original filename for existing automation and notebooks.
    _write_json(output_dir / "optimal_sgd_config.json", optimal)
    _write_json(output_dir / "grid_search_results.json", results)
    _write_csv(output_dir / "trials.csv", summaries)
    print(f"Best candidate: {overall['candidate_id']} (mean score {overall['score_across_setups']:.6g})")
    print(f"Wrote {output_dir / 'optimal_optimizer_config.json'}")
    return output_dir


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in {"grid", "setups", "repeat_seeds", "optimizer"}:
            result[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result["setups"] = result["setups"][:1]
    result["repeat_seeds"] = result["repeat_seeds"][:1]
    result["grid"] = {path: [values[0]] for path, values in result["grid"].items()}
    result["model"]["depth"] = 2
    result["model"]["width"] = 8
    result["training"]["steps"] = 2
    result["training"]["device"] = "cpu"
    data = result["setups"][0]["data"]
    data.update(
        train_interior=8,
        train_boundary=8,
        validation_interior=8,
        validation_boundary=8,
        evaluation_resolution=5,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for the default config")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny two-step trial")
    parser.add_argument(
        "--write-default-config",
        type=Path,
        metavar="PATH",
        help="Write an editable JSON config and exit",
    )
    return parser.parse_args()


def main() -> Path | None:
    args = parse_args()
    if args.write_default_config is not None:
        args.write_default_config.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.write_default_config, DEFAULT_GRID_SEARCH_CONFIG)
        print(f"Wrote {args.write_default_config}")
        return None
    config = copy.deepcopy(DEFAULT_GRID_SEARCH_CONFIG)
    if args.config is not None:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(overrides, Mapping):
            raise TypeError("The config JSON root must be an object")
        config = _deep_merge(config, overrides)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.smoke_test:
        config = _smoke_config(config)
    return run_grid_search(config)


if __name__ == "__main__":
    main()
