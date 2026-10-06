"""Physics-informed utilities for the unit-square Poisson problem."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import torch
from torch import nn

from .average_sharpness import AverageSharpnessResult, evaluate_average_sharpness_closure
from .config import training_config
from .pinn_objective import (
    configure_selective_pinn_loss,
    resolve_sharpness_components,
)
from .pinn_batching import PINNBatcher
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


@dataclass(frozen=True)
class PoissonForcingEvaluation:
    """Metrics for a fixed model evaluated against one scaled Poisson RHS."""

    forcing_scale: float
    rhs_sine_coefficient: float
    rhs_rmse: float
    rhs_change_rmse_from_nominal: float
    nominal_normalized_l2_error: float
    pde_residual_rmse: float
    nominal_normalized_l2_error_change_from_nominal: float
    pde_residual_rmse_change_from_nominal: float


@dataclass(frozen=True)
class PoissonForcingSensitivity:
    """Sensitivity of a fixed model to multiplicative forcing perturbations."""

    nominal_forcing_scale: float
    evaluations: tuple[PoissonForcingEvaluation, ...]
    mean_nominal_normalized_l2_error: float
    worst_nominal_normalized_l2_error: float
    mean_pde_residual_rmse: float
    worst_pde_residual_rmse: float
    max_nominal_normalized_l2_error_change_per_unit_scale: float
    max_pde_residual_rmse_change_per_unit_scale: float


@dataclass(frozen=True)
class LaplaceMetric:
    """First-order scalar metric uncertainty under a local Laplace posterior."""

    central: float
    standard_deviation: float
    confidence_low: float
    confidence_high: float


@dataclass(frozen=True)
class PoissonLaplaceForcingEvaluation:
    """Laplace uncertainty for one alternative Poisson right-hand side."""

    forcing_scale: float
    nominal_normalized_l2_error: LaplaceMetric
    pde_residual_rmse: LaplaceMetric


@dataclass(frozen=True)
class PoissonLaplaceSensitivity:
    """Diagonal-GGN Laplace approximation around fixed final parameters."""

    approximation: str
    prior_precision: float
    likelihood_precision: float
    curvature_probes: int
    confidence_level: float
    seed: int
    parameter_count: int
    clean_weighted_loss: float
    clean_gradient_norm: float
    effective_parameter_count: float
    posterior_parameter_std_mean: float
    posterior_parameter_std_max: float
    evaluations: tuple[PoissonLaplaceForcingEvaluation, ...]


def exact_poisson_solution(coordinates: torch.Tensor) -> torch.Tensor:
    """Return ``sin(pi*x) sin(pi*y)`` at ``(x, y)`` coordinates."""

    return torch.sin(math.pi * coordinates[:, 0]) * torch.sin(
        math.pi * coordinates[:, 1]
    )


def _forcing_scale(value: float) -> float:
    scale = float(value)
    if not math.isfinite(scale):
        raise ValueError("forcing_scale must be finite")
    return scale


def poisson_forcing(
    coordinates: torch.Tensor,
    *,
    forcing_scale: float = 1.0,
) -> torch.Tensor:
    """Return the scaled forcing in ``-Delta u = forcing_scale * f``."""

    return (
        _forcing_scale(forcing_scale)
        * 2.0
        * math.pi**2
        * exact_poisson_solution(coordinates)
    )


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
    forcing_scale: float = 1.0,
) -> torch.Tensor:
    """Compute ``-Delta u_theta - forcing_scale * f`` using autograd."""

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
        return -laplacian - poisson_forcing(
            points,
            forcing_scale=forcing_scale,
        )


def poisson_loss_components(
    model: nn.Module,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
    forcing_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return the complete weighted objective and its unweighted components."""

    residual = poisson_residual(
        model,
        interior_points,
        create_graph=True,
        forcing_scale=forcing_scale,
    )
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
    point_groups = {
        "interior": points.interior.to(device=device, dtype=dtype),
        "boundary": points.boundary.to(device=device, dtype=dtype),
    }
    batcher = PINNBatcher(point_groups, resolved)
    pde_weight, boundary_weight = _pinn_options(config)
    sharpness_components = resolve_sharpness_components(config)

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
    for name in point_groups:
        history[f"{name}_batch_size"] = []
    resolved["effective_batch_sizes"] = dict(batcher.batch_sizes)
    result = TrainingResult(model=model, history=history, config=resolved)
    if resolved["algorithm"] == "s_sam":
        print(
            "S-SAM PINN sharpness components: "
            + ", ".join(sorted(sharpness_components))
        )

    for step in range(steps):
        batches = batcher.next()

        # The batch is selected once per update and captured by the closure, so
        # every clean and perturbed S-SAM evaluation uses identical points.
        def closure() -> dict[str, torch.Tensor]:
            return configure_selective_pinn_loss(
                poisson_loss_components(
                    model,
                    batches["interior"],
                    batches["boundary"],
                    pde_weight=pde_weight,
                    boundary_weight=boundary_weight,
                ),
                pde_weight=pde_weight,
                boundary_weight=boundary_weight,
                sharpness_components=sharpness_components,
            )

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
            ) if "pde" in sharpness_components else float("nan"),
            "gaussian_boundary_loss": outcome.regularized_components.get(
                "boundary_loss", float("nan")
            ) if "boundary" in sharpness_components else float("nan"),
            "learning_rate": outcome.learning_rate,
            "sharpness_scale": scale,
        }
        for name, batch in batches.items():
            record[f"{name}_batch_size"] = int(batch.shape[0])
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
    forcing_scale: float = 1.0,
) -> PoissonMetrics:
    """Evaluate against ``-Delta u = forcing_scale * f``.

    Because the boundary condition is homogeneous and the PDE is linear, the
    exact solution for the scaled forcing is ``forcing_scale * u*``. Model
    parameters are never changed by this evaluation.
    """

    forcing_scale = _forcing_scale(forcing_scale)

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
            exact = forcing_scale * exact_poisson_solution(evaluation)
            relative_l2 = torch.linalg.vector_norm(prediction - exact) / torch.linalg.vector_norm(
                exact
            ).clamp_min(torch.finfo(exact.dtype).eps)
            boundary_rmse = _scalar_prediction(model, boundary).square().mean().sqrt()
        residual_rmse = poisson_residual(
            model,
            evaluation,
            create_graph=False,
            forcing_scale=forcing_scale,
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
                    forcing_scale=forcing_scale,
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


def evaluate_poisson_forcing_sensitivity(
    model: nn.Module,
    points: PoissonPointSets,
    forcing_scales: list[float] | tuple[float, ...],
    *,
    nominal_forcing_scale: float = 1.0,
) -> PoissonForcingSensitivity:
    """Evaluate a fixed model under a sweep of multiplicative RHS changes.

    The model prediction and its Laplacian are evaluated only once. All scale
    results after that are inexpensive tensor arithmetic. The two sensitivity
    values are worst absolute secant slopes away from the nominal scale; lower
    values indicate less metric change per unit RHS-scale perturbation.
    """

    scales = tuple(_forcing_scale(value) for value in forcing_scales)
    nominal_scale = _forcing_scale(nominal_forcing_scale)
    if not scales:
        raise ValueError("forcing_scales must contain at least one value")
    if len(set(scales)) != len(scales):
        raise ValueError("forcing_scales must not contain duplicates")
    if nominal_scale == 0.0:
        raise ValueError(
            "nominal_forcing_scale must be non-zero because it normalizes "
            "the solution error"
        )
    nominal_indices = [
        index
        for index, scale in enumerate(scales)
        if math.isclose(scale, nominal_scale, rel_tol=1e-12, abs_tol=1e-12)
    ]
    if not nominal_indices:
        raise ValueError("forcing_scales must include nominal_forcing_scale")

    parameter = next((value for value in model.parameters() if value.requires_grad), None)
    if parameter is None:
        raise ValueError("The model has no trainable parameters")
    evaluation = points.evaluation.to(device=parameter.device, dtype=parameter.dtype)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = _scalar_prediction(model, evaluation)
            reference_solution = exact_poisson_solution(evaluation)
            nominal_solution_norm = torch.linalg.vector_norm(
                nominal_scale * reference_solution
            ).clamp_min(torch.finfo(reference_solution.dtype).eps)
        # A zero forcing scale exposes the model's fixed left-hand side. It is
        # computed once, then reused for every alternative right-hand side.
        operator_prediction = poisson_residual(
            model,
            evaluation,
            create_graph=False,
            forcing_scale=0.0,
        ).detach()
        reference_forcing = poisson_forcing(evaluation).detach()

        reference_forcing_rmse = float(reference_forcing.square().mean().sqrt())
        raw_metrics: list[tuple[float, float]] = []
        for scale in scales:
            exact = scale * reference_solution
            nominal_normalized_l2 = (
                torch.linalg.vector_norm(prediction - exact)
                / nominal_solution_norm
            )
            residual_rmse = (
                operator_prediction - scale * reference_forcing
            ).square().mean().sqrt()
            raw_metrics.append((float(nominal_normalized_l2), float(residual_rmse)))

        nominal_normalized_l2, nominal_residual_rmse = raw_metrics[
            nominal_indices[0]
        ]
        evaluations = tuple(
            PoissonForcingEvaluation(
                forcing_scale=scale,
                rhs_sine_coefficient=scale * 2.0 * math.pi**2,
                rhs_rmse=abs(scale) * reference_forcing_rmse,
                rhs_change_rmse_from_nominal=(
                    abs(scale - nominal_scale) * reference_forcing_rmse
                ),
                nominal_normalized_l2_error=solution_error,
                pde_residual_rmse=residual_rmse,
                nominal_normalized_l2_error_change_from_nominal=(
                    solution_error - nominal_normalized_l2
                ),
                pde_residual_rmse_change_from_nominal=(
                    residual_rmse - nominal_residual_rmse
                ),
            )
            for scale, (solution_error, residual_rmse) in zip(scales, raw_metrics)
        )
        perturbed = [
            evaluation
            for evaluation in evaluations
            if not math.isclose(
                evaluation.forcing_scale,
                nominal_scale,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]

        def max_change_per_unit_scale(attribute: str) -> float:
            if not perturbed:
                return 0.0
            nominal_value = (
                nominal_normalized_l2
                if attribute == "nominal_normalized_l2_error"
                else nominal_residual_rmse
            )
            return max(
                abs(getattr(value, attribute) - nominal_value)
                / abs(value.forcing_scale - nominal_scale)
                for value in perturbed
            )

        solution_error_values = [value[0] for value in raw_metrics]
        residual_values = [value[1] for value in raw_metrics]
        return PoissonForcingSensitivity(
            nominal_forcing_scale=nominal_scale,
            evaluations=evaluations,
            mean_nominal_normalized_l2_error=(
                sum(solution_error_values) / len(solution_error_values)
            ),
            worst_nominal_normalized_l2_error=max(solution_error_values),
            mean_pde_residual_rmse=sum(residual_values) / len(residual_values),
            worst_pde_residual_rmse=max(residual_values),
            max_nominal_normalized_l2_error_change_per_unit_scale=(
                max_change_per_unit_scale("nominal_normalized_l2_error")
            ),
            max_pde_residual_rmse_change_per_unit_scale=max_change_per_unit_scale(
                "pde_residual_rmse"
            ),
        )
    finally:
        model.train(was_training)


def evaluate_poisson_laplace_sensitivity(
    model: nn.Module,
    points: PoissonPointSets,
    forcing_scales: list[float] | tuple[float, ...],
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
    prior_precision: float = 100.0,
    likelihood_precision: float = 1.0,
    curvature_probes: int = 16,
    confidence_level: float = 0.95,
    seed: int = 2718,
    nominal_forcing_scale: float = 1.0,
) -> PoissonLaplaceSensitivity:
    """Propagate a local diagonal-GGN Laplace posterior to RHS metrics.

    The weighted mean-square PINN loss is treated as twice a Gaussian negative
    log likelihood. A Hutchinson estimator obtains the diagonal of ``J.T @ J``
    without constructing the residual Jacobian. ``prior_precision`` supplies
    the positive diagonal precision needed in flat directions. Scalar metric
    bands use the first-order delta method, so this is a local approximation,
    not a calibrated posterior unless the likelihood and prior precisions have
    themselves been calibrated.
    """

    scales = tuple(_forcing_scale(value) for value in forcing_scales)
    nominal_scale = _forcing_scale(nominal_forcing_scale)
    if not scales:
        raise ValueError("forcing_scales must contain at least one value")
    if len(set(scales)) != len(scales):
        raise ValueError("forcing_scales must not contain duplicates")
    if nominal_scale == 0.0:
        raise ValueError(
            "nominal_forcing_scale must be non-zero because it normalizes "
            "the solution error"
        )
    if pde_weight < 0.0 or boundary_weight < 0.0:
        raise ValueError("PINN loss weights must be non-negative")
    if pde_weight == 0.0 and boundary_weight == 0.0:
        raise ValueError("At least one PINN loss weight must be positive")
    if prior_precision <= 0.0 or not math.isfinite(prior_precision):
        raise ValueError("prior_precision must be positive and finite")
    if likelihood_precision <= 0.0 or not math.isfinite(likelihood_precision):
        raise ValueError("likelihood_precision must be positive and finite")
    if curvature_probes < 1:
        raise ValueError("curvature_probes must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters")
    parameter = parameters[0]
    device = parameter.device
    dtype = parameter.dtype
    interior = points.interior.to(device=device, dtype=dtype)
    boundary = points.boundary.to(device=device, dtype=dtype)
    evaluation = points.evaluation.to(device=device, dtype=dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    was_training = model.training
    model.eval()

    def random_signs_like(values: torch.Tensor) -> torch.Tensor:
        signs = torch.randint(
            0,
            2,
            values.shape,
            generator=generator,
            device="cpu",
            dtype=torch.int64,
        )
        return signs.to(device=values.device, dtype=values.dtype).mul_(2).sub_(1)

    try:
        residual = poisson_residual(
            model,
            interior,
            create_graph=True,
            forcing_scale=1.0,
        )
        boundary_prediction = _scalar_prediction(model, boundary)
        observation_parts = []
        if pde_weight > 0.0:
            observation_parts.append(
                math.sqrt(pde_weight / residual.numel()) * residual
            )
        if boundary_weight > 0.0:
            observation_parts.append(
                math.sqrt(boundary_weight / boundary_prediction.numel())
                * boundary_prediction
            )
        observations = torch.cat(observation_parts)
        half_clean_loss = 0.5 * observations.square().sum()
        clean_gradients = torch.autograd.grad(
            half_clean_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        clean_gradient_squared_norm = sum(
            float(gradient.detach().square().sum())
            for gradient in clean_gradients
            if gradient is not None
        )

        curvature_diagonal = [torch.zeros_like(value) for value in parameters]
        for probe_index in range(curvature_probes):
            projection = (observations * random_signs_like(observations)).sum()
            projected_gradients = torch.autograd.grad(
                projection,
                parameters,
                retain_graph=probe_index + 1 < curvature_probes,
                allow_unused=True,
            )
            with torch.no_grad():
                for estimate, gradient in zip(
                    curvature_diagonal,
                    projected_gradients,
                ):
                    if gradient is not None:
                        estimate.add_(gradient.detach().square())
        curvature_diagonal = [
            value / curvature_probes for value in curvature_diagonal
        ]
        posterior_variances = [
            (
                likelihood_precision * curvature + prior_precision
            ).reciprocal()
            for curvature in curvature_diagonal
        ]
        posterior_standard_deviations = torch.cat(
            [variance.sqrt().reshape(-1) for variance in posterior_variances]
        )
        effective_parameter_count = sum(
            float(
                (
                    likelihood_precision
                    * curvature
                    / (likelihood_precision * curvature + prior_precision)
                ).sum()
            )
            for curvature in curvature_diagonal
        )

        prediction = _scalar_prediction(model, evaluation)
        operator_prediction = poisson_residual(
            model,
            evaluation,
            create_graph=True,
            forcing_scale=0.0,
        )
        reference_solution = exact_poisson_solution(evaluation)
        reference_forcing = poisson_forcing(evaluation)
        nominal_solution_rms = (
            nominal_scale * reference_solution
        ).square().mean().sqrt()
        confidence_z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        tiny = torch.finfo(dtype).tiny

        def metric_summary(metric: torch.Tensor) -> LaplaceMetric:
            gradients = torch.autograd.grad(
                metric,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            variance = sum(
                float((gradient.detach().square() * parameter_variance).sum())
                for gradient, parameter_variance in zip(
                    gradients,
                    posterior_variances,
                )
                if gradient is not None
            )
            central = float(metric.detach())
            standard_deviation = math.sqrt(max(0.0, variance))
            radius = confidence_z * standard_deviation
            return LaplaceMetric(
                central=central,
                standard_deviation=standard_deviation,
                confidence_low=max(0.0, central - radius),
                confidence_high=central + radius,
            )

        evaluations = []
        for scale in scales:
            exact = scale * reference_solution
            difference_mse = (prediction - exact).square().mean()
            nominal_normalized_l2 = (
                difference_mse.clamp_min(tiny).sqrt() / nominal_solution_rms
            )
            pde_residual = operator_prediction - scale * reference_forcing
            residual_rmse = pde_residual.square().mean().clamp_min(tiny).sqrt()
            evaluations.append(
                PoissonLaplaceForcingEvaluation(
                    forcing_scale=scale,
                    nominal_normalized_l2_error=metric_summary(
                        nominal_normalized_l2
                    ),
                    pde_residual_rmse=metric_summary(residual_rmse),
                )
            )

        return PoissonLaplaceSensitivity(
            approximation="diagonal_generalized_gauss_newton_laplace",
            prior_precision=float(prior_precision),
            likelihood_precision=float(likelihood_precision),
            curvature_probes=int(curvature_probes),
            confidence_level=float(confidence_level),
            seed=int(seed),
            parameter_count=sum(value.numel() for value in parameters),
            clean_weighted_loss=float(2.0 * half_clean_loss.detach()),
            clean_gradient_norm=math.sqrt(clean_gradient_squared_norm),
            effective_parameter_count=effective_parameter_count,
            posterior_parameter_std_mean=float(
                posterior_standard_deviations.mean()
            ),
            posterior_parameter_std_max=float(
                posterior_standard_deviations.max()
            ),
            evaluations=tuple(evaluations),
        )
    finally:
        model.train(was_training)
