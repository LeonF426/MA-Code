"""Linear diagonal PINN benchmarks with differentiable fixed feature maps."""

from __future__ import annotations

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
class SpaceTimePointSets:
    """Fixed points for a problem on a one-dimensional space-time domain."""

    interior: torch.Tensor
    initial: torch.Tensor
    boundary: torch.Tensor
    evaluation: torch.Tensor
    space_resolution: int
    time_resolution: int


@dataclass(frozen=True)
class PolynomialPoissonPointSets:
    """Fixed points for the one-dimensional polynomial Poisson problem."""

    interior: torch.Tensor
    boundary: torch.Tensor
    evaluation: torch.Tensor
    evaluation_resolution: int


@dataclass(frozen=True)
class LinearPinnMetrics:
    """Common accuracy, residual, condition, and sharpness measurements."""

    relative_l2_error: float
    pde_residual_rmse: float
    boundary_rmse: float
    average_sharpness: AverageSharpnessResult | None = None


FeatureMap = Callable[[torch.Tensor], torch.Tensor]
ResidualFunction = Callable[[nn.Module, torch.Tensor], torch.Tensor]
LossClosureFactory = Callable[
    [torch.device, torch.dtype], Callable[[], dict[str, torch.Tensor]]
]


def advection_features(coordinates: torch.Tensor) -> torch.Tensor:
    """Return ``(1, x, t)`` for the affine advection solution space."""

    return torch.stack(
        (
            torch.ones_like(coordinates[:, 0]),
            coordinates[:, 0],
            coordinates[:, 1],
        ),
        dim=1,
    )


def polynomial_poisson_features(coordinates: torch.Tensor) -> torch.Tensor:
    """Return ``(1, x, x^2)`` for the quadratic Poisson solution space."""

    x = coordinates[:, 0]
    return torch.stack((torch.ones_like(x), x, x.square()), dim=1)


def heat_features(coordinates: torch.Tensor) -> torch.Tensor:
    """Return ``(x^2, t)`` for the polynomial heat solution space."""

    x = coordinates[:, 0]
    t = coordinates[:, 1]
    return torch.stack((x.square(), t), dim=1)


def exact_advection_solution(coordinates: torch.Tensor) -> torch.Tensor:
    """Return the exact solution ``2 + 3*x + 3*t``."""

    return 2.0 + 3.0 * coordinates[:, 0] + 3.0 * coordinates[:, 1]


def exact_polynomial_poisson_solution(coordinates: torch.Tensor) -> torch.Tensor:
    """Return the exact solution ``1 + 3*x + 2*x^2``."""

    x = coordinates[:, 0]
    return 1.0 + 3.0 * x + 2.0 * x.square()


def exact_heat_solution(coordinates: torch.Tensor) -> torch.Tensor:
    """Return the exact solution ``x^2 + 2*t``."""

    return coordinates[:, 0].square() + 2.0 * coordinates[:, 1]


def _scalar_feature_prediction(
    model: nn.Module,
    coordinates: torch.Tensor,
    feature_map: FeatureMap,
) -> torch.Tensor:
    prediction = model(feature_map(coordinates))
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if prediction.ndim != 1 or prediction.shape[0] != coordinates.shape[0]:
        raise ValueError("A scalar PINN must produce one value per coordinate")
    return prediction


def advection_residual(
    model: nn.Module,
    coordinates: torch.Tensor,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Compute ``u_t - u_x`` for the affine advection benchmark."""

    with torch.enable_grad():
        points = coordinates.detach().clone().requires_grad_(True)
        prediction = _scalar_feature_prediction(model, points, advection_features)
        gradient = torch.autograd.grad(
            prediction,
            points,
            grad_outputs=torch.ones_like(prediction),
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return gradient[:, 1] - gradient[:, 0]


def polynomial_poisson_residual(
    model: nn.Module,
    coordinates: torch.Tensor,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Compute ``-u_xx + 4`` for ``-u_xx = -4``."""

    with torch.enable_grad():
        points = coordinates.detach().clone().requires_grad_(True)
        prediction = _scalar_feature_prediction(
            model, points, polynomial_poisson_features
        )
        first_derivative = torch.autograd.grad(
            prediction,
            points,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0]
        second_derivative = torch.autograd.grad(
            first_derivative,
            points,
            grad_outputs=torch.ones_like(first_derivative),
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0][:, 0]
        return -second_derivative + 4.0


def heat_residual(
    model: nn.Module,
    coordinates: torch.Tensor,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Compute ``u_t - u_xx`` for the polynomial heat benchmark."""

    with torch.enable_grad():
        points = coordinates.detach().clone().requires_grad_(True)
        prediction = _scalar_feature_prediction(model, points, heat_features)
        gradient = torch.autograd.grad(
            prediction,
            points,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
        )[0]
        second_derivative = torch.autograd.grad(
            gradient[:, 0],
            points,
            grad_outputs=torch.ones_like(gradient[:, 0]),
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0][:, 0]
        return gradient[:, 1] - second_derivative


def _condition_loss(
    model: nn.Module,
    coordinate_sets: tuple[torch.Tensor, ...],
    feature_map: FeatureMap,
    exact_solution: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    errors = []
    for coordinates in coordinate_sets:
        prediction = _scalar_feature_prediction(model, coordinates, feature_map)
        errors.append(prediction - exact_solution(coordinates))
    return torch.cat(errors).square().mean()


def advection_loss_components(
    model: nn.Module,
    interior_points: torch.Tensor,
    initial_points: torch.Tensor,
    boundary_points: torch.Tensor,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return weighted advection loss and unweighted diagnostic components."""

    pde_loss = advection_residual(model, interior_points).square().mean()
    boundary_loss = _condition_loss(
        model,
        (initial_points, boundary_points),
        advection_features,
        exact_advection_solution,
    )
    return {
        "loss": pde_weight * pde_loss + boundary_weight * boundary_loss,
        "pde_loss": pde_loss,
        "boundary_loss": boundary_loss,
    }


def polynomial_poisson_loss_components(
    model: nn.Module,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return weighted polynomial-Poisson loss and diagnostic components."""

    pde_loss = polynomial_poisson_residual(model, interior_points).square().mean()
    boundary_loss = _condition_loss(
        model,
        (boundary_points,),
        polynomial_poisson_features,
        exact_polynomial_poisson_solution,
    )
    return {
        "loss": pde_weight * pde_loss + boundary_weight * boundary_loss,
        "pde_loss": pde_loss,
        "boundary_loss": boundary_loss,
    }


def heat_loss_components(
    model: nn.Module,
    interior_points: torch.Tensor,
    initial_points: torch.Tensor,
    boundary_points: torch.Tensor,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return weighted heat-equation loss and diagnostic components."""

    pde_loss = heat_residual(model, interior_points).square().mean()
    boundary_loss = _condition_loss(
        model,
        (initial_points, boundary_points),
        heat_features,
        exact_heat_solution,
    )
    return {
        "loss": pde_weight * pde_loss + boundary_weight * boundary_loss,
        "pde_loss": pde_loss,
        "boundary_loss": boundary_loss,
    }


def _point_dtype(data: Mapping[str, Any]) -> torch.dtype:
    dtype_name = str(data.get("dtype", "float32")).lower()
    dtypes = {"float32": torch.float32, "float64": torch.float64}
    if dtype_name not in dtypes:
        raise ValueError("PINN point dtype must be 'float32' or 'float64'")
    return dtypes[dtype_name]


def _space_time_evaluation(
    space_resolution: int,
    time_resolution: int,
    time_max: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_axis = torch.linspace(0.0, 1.0, space_resolution, dtype=dtype)
    t_axis = torch.linspace(0.0, time_max, time_resolution, dtype=dtype)
    grid_x, grid_t = torch.meshgrid(x_axis, t_axis, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_t.reshape(-1)), dim=1)


def build_advection_points(config: Mapping[str, Any]) -> SpaceTimePointSets:
    """Build fixed collocation, condition, and evaluation points."""

    data = config.get("data", config)
    interior_count = int(data.get("interior_points", 512))
    initial_count = int(data.get("initial_points", 256))
    boundary_count = int(data.get("boundary_points", 256))
    space_resolution = int(data.get("space_resolution", 51))
    time_resolution = int(data.get("time_resolution", 51))
    time_max = float(data.get("time_max", 1.0))
    if min(interior_count, initial_count, boundary_count) < 1:
        raise ValueError("All advection point counts must be positive")
    if min(space_resolution, time_resolution) < 3 or time_max <= 0.0:
        raise ValueError("Resolutions must be at least 3 and time_max positive")
    dtype = _point_dtype(data)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(data.get("seed", 0)))

    interior = torch.rand((interior_count, 2), generator=generator, dtype=dtype)
    interior[:, 1] *= time_max
    initial = torch.rand((initial_count, 2), generator=generator, dtype=dtype)
    initial[:, 1] = 0.0
    boundary = torch.rand((boundary_count, 2), generator=generator, dtype=dtype)
    boundary[:, 0] = 1.0
    boundary[:, 1] *= time_max
    evaluation = _space_time_evaluation(
        space_resolution, time_resolution, time_max, dtype
    )
    return SpaceTimePointSets(
        interior,
        initial,
        boundary,
        evaluation,
        space_resolution,
        time_resolution,
    )


def build_polynomial_poisson_points(
    config: Mapping[str, Any],
) -> PolynomialPoissonPointSets:
    """Build fixed points for the one-dimensional polynomial Poisson problem."""

    data = config.get("data", config)
    interior_count = int(data.get("interior_points", 512))
    boundary_count = int(data.get("boundary_points", 256))
    resolution = int(data.get("evaluation_resolution", 201))
    if interior_count < 1 or boundary_count < 2 or resolution < 3:
        raise ValueError(
            "interior_points must be positive, boundary_points at least 2, "
            "and evaluation_resolution at least 3"
        )
    dtype = _point_dtype(data)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(data.get("seed", 0)))
    interior = torch.rand((interior_count, 1), generator=generator, dtype=dtype)
    boundary = torch.zeros((boundary_count, 1), dtype=dtype)
    boundary[1::2, 0] = 1.0
    evaluation = torch.linspace(0.0, 1.0, resolution, dtype=dtype).unsqueeze(1)
    return PolynomialPoissonPointSets(interior, boundary, evaluation, resolution)


def build_heat_points(config: Mapping[str, Any]) -> SpaceTimePointSets:
    """Build fixed collocation, initial, boundary, and evaluation heat points."""

    data = config.get("data", config)
    interior_count = int(data.get("interior_points", 512))
    initial_count = int(data.get("initial_points", 256))
    boundary_count = int(data.get("boundary_points", 256))
    space_resolution = int(data.get("space_resolution", 51))
    time_resolution = int(data.get("time_resolution", 51))
    time_max = float(data.get("time_max", 1.0))
    if min(interior_count, initial_count) < 1 or boundary_count < 2:
        raise ValueError(
            "interior_points and initial_points must be positive and "
            "boundary_points at least 2"
        )
    if min(space_resolution, time_resolution) < 3 or time_max <= 0.0:
        raise ValueError("Resolutions must be at least 3 and time_max positive")
    dtype = _point_dtype(data)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(data.get("seed", 0)))

    interior = torch.rand((interior_count, 2), generator=generator, dtype=dtype)
    interior[:, 1] *= time_max
    initial = torch.rand((initial_count, 2), generator=generator, dtype=dtype)
    initial[:, 1] = 0.0
    boundary = torch.rand((boundary_count, 2), generator=generator, dtype=dtype)
    boundary[:, 1] *= time_max
    boundary[0::2, 0] = 0.0
    boundary[1::2, 0] = 1.0
    evaluation = _space_time_evaluation(
        space_resolution, time_resolution, time_max, dtype
    )
    return SpaceTimePointSets(
        interior,
        initial,
        boundary,
        evaluation,
        space_resolution,
        time_resolution,
    )


def _pinn_weights(config: Mapping[str, Any]) -> tuple[float, float]:
    options = config.get("pinn", {})
    pde_weight = float(options.get("pde_weight", 1.0))
    boundary_weight = float(options.get("boundary_weight", 1.0))
    if pde_weight < 0.0 or boundary_weight < 0.0:
        raise ValueError("PINN loss weights must be non-negative")
    if pde_weight == 0.0 and boundary_weight == 0.0:
        raise ValueError("At least one PINN loss weight must be positive")
    return pde_weight, boundary_weight


def _train_fixed_pinn(
    model: nn.Module,
    config: Mapping[str, Any],
    closure_factory: LossClosureFactory,
    callbacks: list[Callable[[int, nn.Module, dict[str, Any]], None]] | None,
) -> TrainingResult:
    resolved = training_config(config)
    resolved["loss_requires_grad"] = True
    steps = int(resolved.get("steps", 0))
    if steps < 1:
        raise ValueError("PINN training requires training.steps to be positive")

    torch.manual_seed(int(resolved.get("seed", 0)))
    device = _device(str(resolved.get("device", "auto")))
    model.to(device)
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration as exc:
        raise ValueError("The model has no trainable parameters") from exc
    closure = closure_factory(device, dtype)

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
                float("nan")
                if outcome.regularized_loss is None
                else outcome.regularized_loss
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


def train_advection_pinn(
    model: nn.Module,
    points: SpaceTimePointSets,
    config: Mapping[str, Any],
    callbacks: list[Callable[[int, nn.Module, dict[str, Any]], None]] | None = None,
) -> TrainingResult:
    """Train the affine advection PINN with a fixed set of points."""

    pde_weight, boundary_weight = _pinn_weights(config)

    def factory(device: torch.device, dtype: torch.dtype):
        interior = points.interior.to(device=device, dtype=dtype)
        initial = points.initial.to(device=device, dtype=dtype)
        boundary = points.boundary.to(device=device, dtype=dtype)

        def closure() -> dict[str, torch.Tensor]:
            return advection_loss_components(
                model,
                interior,
                initial,
                boundary,
                pde_weight=pde_weight,
                boundary_weight=boundary_weight,
            )

        return closure

    return _train_fixed_pinn(model, config, factory, callbacks)


def train_polynomial_poisson_pinn(
    model: nn.Module,
    points: PolynomialPoissonPointSets,
    config: Mapping[str, Any],
    callbacks: list[Callable[[int, nn.Module, dict[str, Any]], None]] | None = None,
) -> TrainingResult:
    """Train the quadratic Poisson PINN with a fixed set of points."""

    pde_weight, boundary_weight = _pinn_weights(config)

    def factory(device: torch.device, dtype: torch.dtype):
        interior = points.interior.to(device=device, dtype=dtype)
        boundary = points.boundary.to(device=device, dtype=dtype)

        def closure() -> dict[str, torch.Tensor]:
            return polynomial_poisson_loss_components(
                model,
                interior,
                boundary,
                pde_weight=pde_weight,
                boundary_weight=boundary_weight,
            )

        return closure

    return _train_fixed_pinn(model, config, factory, callbacks)


def train_heat_pinn(
    model: nn.Module,
    points: SpaceTimePointSets,
    config: Mapping[str, Any],
    callbacks: list[Callable[[int, nn.Module, dict[str, Any]], None]] | None = None,
) -> TrainingResult:
    """Train the polynomial heat PINN with a fixed set of points."""

    pde_weight, boundary_weight = _pinn_weights(config)

    def factory(device: torch.device, dtype: torch.dtype):
        interior = points.interior.to(device=device, dtype=dtype)
        initial = points.initial.to(device=device, dtype=dtype)
        boundary = points.boundary.to(device=device, dtype=dtype)

        def closure() -> dict[str, torch.Tensor]:
            return heat_loss_components(
                model,
                interior,
                initial,
                boundary,
                pde_weight=pde_weight,
                boundary_weight=boundary_weight,
            )

        return closure

    return _train_fixed_pinn(model, config, factory, callbacks)


def _evaluate_fixed_pinn(
    model: nn.Module,
    evaluation_points: torch.Tensor,
    interior_points: torch.Tensor,
    condition_points: tuple[torch.Tensor, ...],
    feature_map: FeatureMap,
    exact_solution: Callable[[torch.Tensor], torch.Tensor],
    residual_function: Callable[..., torch.Tensor],
    loss_closure: Callable[
        [torch.Tensor, tuple[torch.Tensor, ...]], dict[str, torch.Tensor]
    ],
    *,
    sharpness_scale: float | None,
    sharpness_samples: int,
    sharpness_seed: int,
    antithetic: bool,
    normalized: bool,
) -> LinearPinnMetrics:
    parameter = next((value for value in model.parameters() if value.requires_grad), None)
    if parameter is None:
        raise ValueError("The model has no trainable parameters")
    evaluation = evaluation_points.to(device=parameter.device, dtype=parameter.dtype)
    interior = interior_points.to(device=parameter.device, dtype=parameter.dtype)
    conditions = tuple(
        points.to(device=parameter.device, dtype=parameter.dtype)
        for points in condition_points
    )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = _scalar_feature_prediction(model, evaluation, feature_map)
            exact = exact_solution(evaluation)
            relative_l2 = torch.linalg.vector_norm(
                prediction - exact
            ) / torch.linalg.vector_norm(exact).clamp_min(
                torch.finfo(exact.dtype).eps
            )
            condition_errors = [
                _scalar_feature_prediction(model, points, feature_map)
                - exact_solution(points)
                for points in conditions
            ]
            boundary_rmse = torch.cat(condition_errors).square().mean().sqrt()
        residual_rmse = (
            residual_function(model, evaluation, create_graph=False)
            .detach()
            .square()
            .mean()
            .sqrt()
        )

        sharpness = None
        if sharpness_scale is not None:

            def closure() -> torch.Tensor:
                return loss_closure(interior, conditions)["loss"]

            sharpness = evaluate_average_sharpness_closure(
                model,
                closure,
                sharpness_scale,
                samples=sharpness_samples,
                seed=sharpness_seed,
                antithetic=antithetic,
                normalized=normalized,
                requires_grad=True,
            )
        return LinearPinnMetrics(
            relative_l2_error=float(relative_l2),
            pde_residual_rmse=float(residual_rmse),
            boundary_rmse=float(boundary_rmse),
            average_sharpness=sharpness,
        )
    finally:
        model.train(was_training)


def evaluate_advection_pinn(
    model: nn.Module,
    points: SpaceTimePointSets,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
    sharpness_scale: float | None = None,
    sharpness_samples: int = 128,
    sharpness_seed: int = 12345,
    antithetic: bool = True,
    normalized: bool = False,
) -> LinearPinnMetrics:
    """Evaluate the affine advection model and optional average sharpness."""

    def loss(interior, conditions):
        return advection_loss_components(
            model,
            interior,
            conditions[0],
            conditions[1],
            pde_weight=pde_weight,
            boundary_weight=boundary_weight,
        )

    return _evaluate_fixed_pinn(
        model,
        points.evaluation,
        points.interior,
        (points.initial, points.boundary),
        advection_features,
        exact_advection_solution,
        advection_residual,
        loss,
        sharpness_scale=sharpness_scale,
        sharpness_samples=sharpness_samples,
        sharpness_seed=sharpness_seed,
        antithetic=antithetic,
        normalized=normalized,
    )


def evaluate_polynomial_poisson_pinn(
    model: nn.Module,
    points: PolynomialPoissonPointSets,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
    sharpness_scale: float | None = None,
    sharpness_samples: int = 128,
    sharpness_seed: int = 12345,
    antithetic: bool = True,
    normalized: bool = False,
) -> LinearPinnMetrics:
    """Evaluate the quadratic Poisson model and optional average sharpness."""

    def loss(interior, conditions):
        return polynomial_poisson_loss_components(
            model,
            interior,
            conditions[0],
            pde_weight=pde_weight,
            boundary_weight=boundary_weight,
        )

    return _evaluate_fixed_pinn(
        model,
        points.evaluation,
        points.interior,
        (points.boundary,),
        polynomial_poisson_features,
        exact_polynomial_poisson_solution,
        polynomial_poisson_residual,
        loss,
        sharpness_scale=sharpness_scale,
        sharpness_samples=sharpness_samples,
        sharpness_seed=sharpness_seed,
        antithetic=antithetic,
        normalized=normalized,
    )


def evaluate_heat_pinn(
    model: nn.Module,
    points: SpaceTimePointSets,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
    sharpness_scale: float | None = None,
    sharpness_samples: int = 128,
    sharpness_seed: int = 12345,
    antithetic: bool = True,
    normalized: bool = False,
) -> LinearPinnMetrics:
    """Evaluate the polynomial heat model and optional average sharpness."""

    def loss(interior, conditions):
        return heat_loss_components(
            model,
            interior,
            conditions[0],
            conditions[1],
            pde_weight=pde_weight,
            boundary_weight=boundary_weight,
        )

    return _evaluate_fixed_pinn(
        model,
        points.evaluation,
        points.interior,
        (points.initial, points.boundary),
        heat_features,
        exact_heat_solution,
        heat_residual,
        loss,
        sharpness_scale=sharpness_scale,
        sharpness_samples=sharpness_samples,
        sharpness_seed=sharpness_seed,
        antithetic=antithetic,
        normalized=normalized,
    )
