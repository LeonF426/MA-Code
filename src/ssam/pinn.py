"""Physics-informed utilities for the unit-square Poisson problem."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .average_sharpness import AverageSharpnessResult, evaluate_average_sharpness_closure
from .config import training_config
from .schedules import build_learning_rate_policy, build_schedule
from .trainers import TrainingResult, _device, _inferred_policy_shape
from .update_rules import build_update_rule


@dataclass(frozen=True)
class PoissonPointSets:
    """Fixed collocation and evaluation coordinates for a Poisson experiment."""

    interior: torch.Tensor
    boundary: torch.Tensor
    evaluation: torch.Tensor
    evaluation_resolution: int


@dataclass(frozen=True)
class PoissonMetrics:
    relative_l2_error: float
    pde_residual_rmse: float
    boundary_rmse: float
    average_sharpness: AverageSharpnessResult | None = None


def exact_poisson_solution(coordinates: torch.Tensor) -> torch.Tensor:
    """Return ``sin(pi*x) sin(pi*y)`` at ``(x, y)`` coordinates."""

    return torch.sin(math.pi * coordinates[:, 0]) * torch.sin(
        math.pi * coordinates[:, 1]
    )


def poisson_forcing(coordinates: torch.Tensor) -> torch.Tensor:
    """Return the forcing in ``-Delta u = f`` for the reference solution."""

    return 2.0 * math.pi**2 * exact_poisson_solution(coordinates)


def _scalar_prediction(model: nn.Module, coordinates: torch.Tensor) -> torch.Tensor:
    prediction = model(coordinates)
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if prediction.ndim != 1 or prediction.shape[0] != coordinates.shape[0]:
        raise ValueError("A Poisson PINN must produce one scalar per coordinate")
    return prediction


def poisson_residual(
    model: nn.Module,
    coordinates: torch.Tensor,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Compute ``-Delta u_theta - f`` using coordinate autograd."""

    with torch.enable_grad():
        points = coordinates.detach().clone().requires_grad_(True)
        prediction = _scalar_prediction(model, points)
        gradient = torch.autograd.grad(
            prediction,
            points,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
        )[0]
        laplacian = torch.zeros_like(prediction)
        for dimension in range(2):
            second_derivative = torch.autograd.grad(
                gradient[:, dimension],
                points,
                grad_outputs=torch.ones_like(gradient[:, dimension]),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, dimension]
            laplacian = laplacian + second_derivative
        return -laplacian - poisson_forcing(points)


def poisson_loss_components(
    model: nn.Module,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return the complete weighted objective and its unweighted components."""

    residual = poisson_residual(model, interior_points, create_graph=True)
    pde_loss = residual.square().mean()
    boundary_prediction = _scalar_prediction(model, boundary_points)
    boundary_loss = boundary_prediction.square().mean()
    return {
        "loss": pde_weight * pde_loss + boundary_weight * boundary_loss,
        "pde_loss": pde_loss,
        "boundary_loss": boundary_loss,
    }


def build_poisson_points(config: Mapping[str, Any]) -> PoissonPointSets:
    """Build reproducible fixed collocation points from a config or data section."""

    data = config.get("data", config)
    interior_count = int(data.get("interior_points", 512))
    boundary_count = int(data.get("boundary_points", 256))
    resolution = int(data.get("evaluation_resolution", 51))
    if interior_count < 1 or boundary_count < 4 or resolution < 3:
        raise ValueError(
            "interior_points must be positive, boundary_points at least 4, "
            "and evaluation_resolution at least 3"
        )
    dtype_name = str(data.get("dtype", "float32")).lower()
    dtypes = {"float32": torch.float32, "float64": torch.float64}
    if dtype_name not in dtypes:
        raise ValueError("Poisson point dtype must be 'float32' or 'float64'")
    dtype = dtypes[dtype_name]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(data.get("seed", 0)))

    interior = torch.rand((interior_count, 2), generator=generator, dtype=dtype)
    boundary = torch.rand((boundary_count, 2), generator=generator, dtype=dtype)
    sides = torch.arange(boundary_count) % 4
    boundary[sides == 0, 0] = 0.0
    boundary[sides == 1, 0] = 1.0
    boundary[sides == 2, 1] = 0.0
    boundary[sides == 3, 1] = 1.0

    axis = torch.linspace(0.0, 1.0, resolution, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    evaluation = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    return PoissonPointSets(interior, boundary, evaluation, resolution)


def _pinn_options(config: Mapping[str, Any]) -> tuple[float, float]:
    options = config.get("pinn", {})
    pde_weight = float(options.get("pde_weight", 1.0))
    boundary_weight = float(options.get("boundary_weight", 1.0))
    if pde_weight < 0.0 or boundary_weight < 0.0:
        raise ValueError("PINN loss weights must be non-negative")
    if pde_weight == 0.0 and boundary_weight == 0.0:
        raise ValueError("At least one PINN loss weight must be positive")
    return pde_weight, boundary_weight


def train_poisson_pinn(
    model: nn.Module,
    points: PoissonPointSets,
    config: Mapping[str, Any],
    callbacks: list[Callable[[int, nn.Module, dict[str, Any]], None]] | None = None,
) -> TrainingResult:
    """Train on fixed points with SGD/GD or the repository's Gaussian S-SAM."""

    resolved = training_config(config)
    resolved["loss_requires_grad"] = True
    steps = int(resolved.get("steps", 0))
    if steps < 1:
        raise ValueError("Poisson PINN training requires training.steps to be positive")

    torch.manual_seed(int(resolved.get("seed", 0)))
    device = _device(str(resolved.get("device", "auto")))
    model.to(device)
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration as exc:
        raise ValueError("The model has no trainable parameters") from exc
    interior = points.interior.to(device=device, dtype=dtype)
    boundary = points.boundary.to(device=device, dtype=dtype)
    pde_weight, boundary_weight = _pinn_options(config)

    dimension, depth = _inferred_policy_shape(model, config)
    learning_rate_policy = build_learning_rate_policy(
        resolved["learning_rate"],
        default_dimension=dimension,
        default_depth=depth,
    )
    if learning_rate_policy.requires_regularized_loss and resolved["algorithm"] != "s_sam":
        raise ValueError("An objective-dependent learning rate requires algorithm 's_sam'")
    sharpness_schedule = build_schedule(resolved.get("sharpness_scale", 0.0))
    update_rule = build_update_rule(model, resolved)
    callbacks = callbacks or []

    history: dict[str, list[Any]] = {
        "step": [],
        "loss": [],
        "clean_loss": [],
        "regularized_loss": [],
        "clean_pde_loss": [],
        "clean_boundary_loss": [],
        "gaussian_pde_loss": [],
        "gaussian_boundary_loss": [],
        "learning_rate": [],
        "sharpness_scale": [],
    }
    result = TrainingResult(model=model, history=history, config=resolved)

    # These tensors are captured once. Every clean and perturbed closure call in
    # an update therefore uses identical collocation points.
    def closure() -> dict[str, torch.Tensor]:
        return poisson_loss_components(
            model,
            interior,
            boundary,
            pde_weight=pde_weight,
            boundary_weight=boundary_weight,
        )

    for step in range(steps):
        scale = float(sharpness_schedule(step))
        outcome = update_rule.step(
            closure,
            scale,
            step_index=step,
            learning_rate_policy=learning_rate_policy,
        )
        record: dict[str, Any] = {
            "step": step,
            "loss": outcome.loss,
            "clean_loss": outcome.clean_loss,
            "regularized_loss": (
                float("nan") if outcome.regularized_loss is None else outcome.regularized_loss
            ),
            "clean_pde_loss": outcome.clean_components["pde_loss"],
            "clean_boundary_loss": outcome.clean_components["boundary_loss"],
            "gaussian_pde_loss": outcome.regularized_components.get(
                "pde_loss", float("nan")
            ),
            "gaussian_boundary_loss": outcome.regularized_components.get(
                "boundary_loss", float("nan")
            ),
            "learning_rate": outcome.learning_rate,
            "sharpness_scale": scale,
        }
        for name, value in record.items():
            history[name].append(value)
        for callback in callbacks:
            callback(step, model, record)
    return result


def evaluate_poisson_pinn(
    model: nn.Module,
    points: PoissonPointSets,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
    sharpness_scale: float | None = None,
    sharpness_samples: int = 128,
    sharpness_seed: int = 12345,
    antithetic: bool = True,
    normalized: bool = False,
) -> PoissonMetrics:
    """Evaluate solution, residual, boundary, and optional sharpness metrics."""

    parameter = next((value for value in model.parameters() if value.requires_grad), None)
    if parameter is None:
        raise ValueError("The model has no trainable parameters")
    evaluation = points.evaluation.to(device=parameter.device, dtype=parameter.dtype)
    boundary = points.boundary.to(device=parameter.device, dtype=parameter.dtype)
    interior = points.interior.to(device=parameter.device, dtype=parameter.dtype)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = _scalar_prediction(model, evaluation)
            exact = exact_poisson_solution(evaluation)
            relative_l2 = torch.linalg.vector_norm(prediction - exact) / torch.linalg.vector_norm(
                exact
            ).clamp_min(torch.finfo(exact.dtype).eps)
            boundary_rmse = _scalar_prediction(model, boundary).square().mean().sqrt()
        residual_rmse = poisson_residual(
            model, evaluation, create_graph=False
        ).detach().square().mean().sqrt()

        sharpness = None
        if sharpness_scale is not None:
            def loss_closure() -> torch.Tensor:
                return poisson_loss_components(
                    model,
                    interior,
                    boundary,
                    pde_weight=pde_weight,
                    boundary_weight=boundary_weight,
                )["loss"]

            sharpness = evaluate_average_sharpness_closure(
                model,
                loss_closure,
                sharpness_scale,
                samples=sharpness_samples,
                seed=sharpness_seed,
                antithetic=antithetic,
                normalized=normalized,
                requires_grad=True,
            )
        return PoissonMetrics(
            relative_l2_error=float(relative_l2),
            pde_residual_rmse=float(residual_rmse),
            boundary_rmse=float(boundary_rmse),
            average_sharpness=sharpness,
        )
    finally:
        model.train(was_training)
