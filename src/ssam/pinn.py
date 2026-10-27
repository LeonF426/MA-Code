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
    """Metrics for a fixed model evaluated against one alternative Poisson RHS."""

    forcing_scale: float
    rhs_sine_coefficient: float
    rhs_rmse: float
    rhs_change_rmse_from_nominal: float
    nominal_normalized_l2_error: float
    pde_residual_rmse: float
    boundary_condition_rmse: float
    nominal_normalized_l2_error_change_from_nominal: float
    pde_residual_rmse_change_from_nominal: float
    forcing_mode: str = "scaled"
    forcing_perturbation: float = 1.0
    rhs_constant: float = 0.0


@dataclass(frozen=True)
class PoissonForcingSensitivity:
    """Sensitivity of a fixed model to multiplicative or additive RHS changes."""

    nominal_forcing_scale: float
    evaluations: tuple[PoissonForcingEvaluation, ...]
    mean_nominal_normalized_l2_error: float
    worst_nominal_normalized_l2_error: float
    mean_pde_residual_rmse: float
    worst_pde_residual_rmse: float
    max_nominal_normalized_l2_error_change_per_unit_scale: float
    max_pde_residual_rmse_change_per_unit_scale: float
    forcing_mode: str = "scaled"
    nominal_forcing_perturbation: float = 1.0
    additive_solution_terms: int | None = None


@dataclass(frozen=True)
class PoissonBoundaryEvaluation:
    """Metrics for one scaled or additive Dirichlet-boundary perturbation."""

    boundary_mode: str
    boundary_perturbation: float
    boundary_scale: float
    boundary_additive_scale: float
    boundary_value_rmse: float
    boundary_change_rmse_from_nominal: float
    nominal_normalized_l2_error: float
    pde_residual_rmse: float
    boundary_condition_rmse: float
    nominal_normalized_l2_error_change_from_nominal: float
    boundary_condition_rmse_change_from_nominal: float


@dataclass(frozen=True)
class PoissonBoundarySensitivity:
    """Sensitivity of a frozen model to a family of Dirichlet data."""

    boundary_mode: str
    nominal_boundary_perturbation: float
    boundary_condition: dict[str, float | str]
    additive_boundary_condition: dict[str, float | str] | None
    evaluations: tuple[PoissonBoundaryEvaluation, ...]
    mean_nominal_normalized_l2_error: float
    worst_nominal_normalized_l2_error: float
    mean_boundary_condition_rmse: float
    worst_boundary_condition_rmse: float
    pde_residual_rmse: float
    max_nominal_normalized_l2_error_change_per_unit_perturbation: float
    max_boundary_condition_rmse_change_per_unit_perturbation: float


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
    boundary_condition_rmse: LaplaceMetric
    forcing_mode: str = "scaled"
    forcing_perturbation: float = 1.0


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
    forcing_mode: str = "scaled"
    nominal_forcing_perturbation: float = 1.0
    additive_solution_terms: int | None = None


_BOUNDARY_COEFFICIENTS = (
    "constant",
    "x",
    "y",
    "xy",
    "x2_minus_y2",
)
_BOUNDARY_NAMES = {
    "zero": (),
    "constant": ("constant",),
    "affine": ("constant", "x", "y"),
    "bilinear": ("constant", "x", "y", "xy"),
    "harmonic_polynomial": _BOUNDARY_COEFFICIENTS,
}


def resolve_poisson_boundary_condition(
    boundary_condition: str | Mapping[str, Any] | None = None,
) -> dict[str, float | str]:
    """Validate a named harmonic extension used as Dirichlet boundary data.

    Supported choices are ``zero``, ``constant``, ``affine``, ``bilinear``,
    and ``harmonic_polynomial``.  The latter uses the harmonic basis
    ``1, x, y, x*y, x^2-y^2``.  Because every basis function has zero
    Laplacian, adding the extension changes the boundary values and exact
    solution without silently changing the configured Poisson right-hand side.
    """

    if boundary_condition is None:
        options: dict[str, Any] = {"name": "zero"}
    elif isinstance(boundary_condition, str):
        options = {"name": boundary_condition}
    elif isinstance(boundary_condition, Mapping):
        options = dict(boundary_condition)
    else:
        raise TypeError("boundary_condition must be a string, mapping, or None")
    name = str(options.pop("name", options.pop("type", "zero"))).strip().lower()
    if name not in _BOUNDARY_NAMES:
        raise ValueError(
            "boundary_condition.name must be one of "
            f"{sorted(_BOUNDARY_NAMES)}"
        )
    if name == "constant" and "value" in options:
        if "constant" in options:
            raise ValueError("Set only one of boundary value and constant")
        options["constant"] = options.pop("value")
    allowed = set(_BOUNDARY_NAMES[name])
    unknown = set(options) - set(_BOUNDARY_COEFFICIENTS)
    if unknown:
        raise ValueError(
            f"Unknown {name} boundary-condition keys: {sorted(unknown)}"
        )
    disallowed_nonzero = {
        coefficient
        for coefficient in set(options) - allowed
        if float(options[coefficient]) != 0.0
    }
    if disallowed_nonzero:
        raise ValueError(
            f"Boundary choice {name!r} does not support non-zero coefficients: "
            f"{sorted(disallowed_nonzero)}"
        )
    resolved: dict[str, float | str] = {"name": name}
    for coefficient in _BOUNDARY_COEFFICIENTS:
        value = float(options.get(coefficient, 0.0))
        if not math.isfinite(value):
            raise ValueError("Boundary-condition coefficients must be finite")
        resolved[coefficient] = value
    return resolved


def poisson_boundary_values(
    coordinates: torch.Tensor,
    boundary_condition: str | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Evaluate the harmonic extension defining the Dirichlet data ``g``."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    condition = resolve_poisson_boundary_condition(boundary_condition)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    return (
        float(condition["constant"]) * torch.ones_like(x)
        + float(condition["x"]) * x
        + float(condition["y"]) * y
        + float(condition["xy"]) * x * y
        + float(condition["x2_minus_y2"]) * (x.square() - y.square())
    )


def exact_poisson_solution(
    coordinates: torch.Tensor,
    a: float = 1.0,
    *,
    boundary_condition: str | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Return the exact sine solution plus a configured harmonic extension."""

    sine_solution = a * torch.sin(math.pi * coordinates[:, 0]) * torch.sin(
        math.pi * coordinates[:, 1]
    )
    return sine_solution + poisson_boundary_values(
        coordinates,
        boundary_condition,
    )


def constant_poisson_solution(
    coordinates: torch.Tensor,
    *,
    series_terms: int = 50,
) -> torch.Tensor:
    r"""Solve ``-Delta w = 1`` on the unit square with ``w = 0`` at its boundary.

    The solution is evaluated through the double sine series

    ``sum 16 sin(m*pi*x) sin(n*pi*y) / (pi^4*m*n*(m^2+n^2))``

    over the requested number of positive odd modes in each dimension. This is
    the response ``g=1`` needed by the additive forcing family ``f + eps*g``.
    """

    terms = int(series_terms)
    if terms < 1:
        raise ValueError("series_terms must be positive")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    modes = torch.arange(
        1,
        2 * terms,
        2,
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    mode_x = modes[:, None]
    mode_y = modes[None, :]
    coefficients = 16.0 / (
        math.pi**4
        * mode_x
        * mode_y
        * (mode_x.square() + mode_y.square())
    )
    sin_x = torch.sin(math.pi * coordinates[:, :1] * modes)
    sin_y = torch.sin(math.pi * coordinates[:, 1:] * modes)
    return torch.einsum("im,mn,in->i", sin_x, coefficients, sin_y)


def _forcing_scale(value: float) -> float:
    scale = float(value)
    if not math.isfinite(scale):
        raise ValueError("forcing_scale must be finite")
    return scale


def _forcing_additive(value: float) -> float:
    additive = float(value)
    if not math.isfinite(additive):
        raise ValueError("forcing_additive must be finite")
    return additive


def _forcing_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"scaled", "additive"}:
        raise ValueError("forcing_mode must be 'scaled' or 'additive'")
    return mode


def _forcing_parameters(mode: str, perturbation: float) -> tuple[float, float]:
    """Return ``(scale, additive)`` for a configured perturbation value."""

    if mode == "scaled":
        return perturbation, 0.0
    return 1.0, perturbation


def _forcing_perturbation(mode: str, value: float) -> float:
    return _forcing_scale(value) if mode == "scaled" else _forcing_additive(value)


def _exact_solution_for_forcing(
    coordinates: torch.Tensor,
    *,
    forcing_scale: float,
    forcing_additive: float,
    additive_solution_terms: int,
    boundary_condition: str | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    return (
        forcing_scale * exact_poisson_solution(coordinates)
        + forcing_additive
        * constant_poisson_solution(
            coordinates,
            series_terms=additive_solution_terms,
        )
        + poisson_boundary_values(coordinates, boundary_condition)
    )


def poisson_forcing(
    coordinates: torch.Tensor,
    *,
    forcing_scale: float = 1.0,
    forcing_additive: float = 0.0,
) -> torch.Tensor:
    """Return ``forcing_scale * f + forcing_additive * 1``."""

    return (
        _forcing_scale(forcing_scale)
        * 2.0
        * math.pi**2
        * exact_poisson_solution(coordinates)
        + _forcing_additive(forcing_additive) * torch.ones_like(coordinates[:, 0])
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
    forcing_additive: float = 0.0,
) -> torch.Tensor:
    """Compute ``-Delta u_theta - (forcing_scale*f + forcing_additive)``."""

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
            forcing_additive=forcing_additive,
        )


def poisson_loss_components(
    model: nn.Module,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    *,
    pde_weight: float = 1.0,
    boundary_weight: float = 1.0,
    forcing_scale: float = 1.0,
    forcing_additive: float = 0.0,
    boundary_condition: str | Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Return the complete weighted objective and its unweighted components."""

    residual = poisson_residual(
        model,
        interior_points,
        create_graph=True,
        forcing_scale=forcing_scale,
        forcing_additive=forcing_additive,
    )
    pde_loss = residual.square().mean()
    boundary_prediction = _scalar_prediction(model, boundary_points)
    boundary_target = poisson_boundary_values(
        boundary_points,
        boundary_condition,
    )
    boundary_loss = (boundary_prediction - boundary_target).square().mean()
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
    boundary_condition = config.get("pinn", {}).get("boundary_condition")
    # Resolve before the first training step so malformed configurations fail
    # immediately rather than part-way through an expensive run.
    resolve_poisson_boundary_condition(boundary_condition)
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
                    boundary_condition=boundary_condition,
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
    forcing_additive: float = 0.0,
    additive_solution_terms: int = 50,
    boundary_condition: str | Mapping[str, Any] | None = None,
) -> PoissonMetrics:
    """Evaluate against ``-Delta u = forcing_scale*f + forcing_additive``.

    The constant-forcing response used by the additive term is evaluated with
    :func:`constant_poisson_solution`. ``boundary_condition`` supplies an exact
    harmonic Dirichlet extension. Model parameters are never changed.
    """

    forcing_scale = _forcing_scale(forcing_scale)
    forcing_additive = _forcing_additive(forcing_additive)

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
            exact = _exact_solution_for_forcing(
                evaluation,
                forcing_scale=forcing_scale,
                forcing_additive=forcing_additive,
                additive_solution_terms=additive_solution_terms,
                boundary_condition=boundary_condition,
            )
            relative_l2 = torch.linalg.vector_norm(prediction - exact) / torch.linalg.vector_norm(
                exact
            ).clamp_min(torch.finfo(exact.dtype).eps)
            boundary_rmse = (
                _scalar_prediction(model, boundary)
                - poisson_boundary_values(boundary, boundary_condition)
            ).square().mean().sqrt()
        residual_rmse = poisson_residual(
            model,
            evaluation,
            create_graph=False,
            forcing_scale=forcing_scale,
            forcing_additive=forcing_additive,
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
                    forcing_additive=forcing_additive,
                    boundary_condition=boundary_condition,
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
    nominal_forcing_scale: float | None = None,
    forcing_mode: str = "scaled",
    nominal_forcing_perturbation: float | None = None,
    additive_solution_terms: int = 50,
    boundary_condition: str | Mapping[str, Any] | None = None,
) -> PoissonForcingSensitivity:
    """Evaluate a fixed model under scaled ``a*f`` or additive ``f+eps`` RHSs.

    ``forcing_scales`` is retained as the argument name for API compatibility;
    in additive mode its values are epsilon. A configured Dirichlet condition
    remains fixed throughout the sweep. The model prediction and its Laplacian
    are evaluated only once. Sensitivities are worst absolute secant slopes
    away from the configured nominal perturbation.
    """

    mode = _forcing_mode(forcing_mode)
    resolved_boundary = resolve_poisson_boundary_condition(boundary_condition)
    perturbations = tuple(
        _forcing_perturbation(mode, value) for value in forcing_scales
    )
    if nominal_forcing_scale is not None and nominal_forcing_perturbation is not None:
        raise ValueError(
            "Set only one of nominal_forcing_scale and "
            "nominal_forcing_perturbation"
        )
    configured_nominal = (
        nominal_forcing_perturbation
        if nominal_forcing_perturbation is not None
        else nominal_forcing_scale
    )
    nominal_perturbation = _forcing_perturbation(
        mode,
        (1.0 if mode == "scaled" else 0.0)
        if configured_nominal is None
        else configured_nominal
    )
    if not perturbations:
        raise ValueError("forcing_scales must contain at least one value")
    if len(set(perturbations)) != len(perturbations):
        raise ValueError("forcing_scales must not contain duplicates")
    if mode == "scaled" and nominal_perturbation == 0.0:
        raise ValueError(
            "nominal_forcing_scale must be non-zero because it normalizes "
            "the solution error"
        )
    nominal_indices = [
        index
        for index, perturbation in enumerate(perturbations)
        if math.isclose(
            perturbation,
            nominal_perturbation,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ]
    if not nominal_indices:
        raise ValueError(
            "forcing_scales must include nominal forcing perturbation"
        )

    parameter = next((value for value in model.parameters() if value.requires_grad), None)
    if parameter is None:
        raise ValueError("The model has no trainable parameters")
    evaluation = points.evaluation.to(device=parameter.device, dtype=parameter.dtype)
    boundary = points.boundary.to(device=parameter.device, dtype=parameter.dtype)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = _scalar_prediction(model, evaluation)
            boundary_condition_rmse = float(
                (
                    _scalar_prediction(model, boundary)
                    - poisson_boundary_values(boundary, resolved_boundary)
                )
                .square()
                .mean()
                .sqrt()
            )
            reference_solution = exact_poisson_solution(evaluation)
            boundary_solution = poisson_boundary_values(
                evaluation,
                resolved_boundary,
            )
            additive_solution = (
                constant_poisson_solution(
                    evaluation,
                    series_terms=additive_solution_terms,
                )
                if mode == "additive"
                else torch.zeros_like(reference_solution)
            )
            nominal_scale, nominal_additive = _forcing_parameters(
                mode,
                nominal_perturbation,
            )
            nominal_solution = (
                nominal_scale * reference_solution
                + nominal_additive * additive_solution
                + boundary_solution
            )
            nominal_solution_norm = torch.linalg.vector_norm(
                nominal_solution
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

        raw_metrics: list[tuple[float, float, float, float]] = []
        nominal_rhs = poisson_forcing(
            evaluation,
            forcing_scale=nominal_scale,
            forcing_additive=nominal_additive,
        ).detach()
        for perturbation in perturbations:
            scale, additive = _forcing_parameters(mode, perturbation)
            exact = (
                scale * reference_solution
                + additive * additive_solution
                + boundary_solution
            )
            nominal_normalized_l2 = (
                torch.linalg.vector_norm(prediction - exact)
                / nominal_solution_norm
            )
            rhs = scale * reference_forcing + additive
            residual_rmse = (operator_prediction - rhs).square().mean().sqrt()
            rhs_rmse = rhs.square().mean().sqrt()
            rhs_change_rmse = (rhs - nominal_rhs).square().mean().sqrt()
            raw_metrics.append(
                (
                    float(nominal_normalized_l2),
                    float(residual_rmse),
                    float(rhs_rmse),
                    float(rhs_change_rmse),
                )
            )

        nominal_normalized_l2, nominal_residual_rmse, _, _ = raw_metrics[
            nominal_indices[0]
        ]
        evaluation_values = []
        for perturbation, raw_metric in zip(perturbations, raw_metrics):
            solution_error, residual_rmse, rhs_rmse, rhs_change_rmse = raw_metric
            scale, additive = _forcing_parameters(mode, perturbation)
            evaluation_values.append(
                PoissonForcingEvaluation(
                    forcing_mode=mode,
                    forcing_perturbation=perturbation,
                    forcing_scale=scale,
                    rhs_sine_coefficient=scale * 2.0 * math.pi**2,
                    rhs_constant=additive,
                    rhs_rmse=rhs_rmse,
                    rhs_change_rmse_from_nominal=rhs_change_rmse,
                    nominal_normalized_l2_error=solution_error,
                    pde_residual_rmse=residual_rmse,
                    boundary_condition_rmse=boundary_condition_rmse,
                    nominal_normalized_l2_error_change_from_nominal=(
                        solution_error - nominal_normalized_l2
                    ),
                    pde_residual_rmse_change_from_nominal=(
                        residual_rmse - nominal_residual_rmse
                    ),
                )
            )
        evaluations = tuple(evaluation_values)
        perturbed = [
            evaluation
            for evaluation in evaluations
            if not math.isclose(
                evaluation.forcing_perturbation,
                nominal_perturbation,
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
                / abs(value.forcing_perturbation - nominal_perturbation)
                for value in perturbed
            )

        solution_error_values = [value[0] for value in raw_metrics]
        residual_values = [value[1] for value in raw_metrics]
        return PoissonForcingSensitivity(
            nominal_forcing_scale=nominal_scale,
            forcing_mode=mode,
            nominal_forcing_perturbation=nominal_perturbation,
            additive_solution_terms=(
                int(additive_solution_terms) if mode == "additive" else None
            ),
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


def evaluate_poisson_boundary_sensitivity(
    model: nn.Module,
    points: PoissonPointSets,
    perturbation_values: list[float] | tuple[float, ...],
    *,
    boundary_condition: str | Mapping[str, Any] | None = None,
    boundary_mode: str = "scaled",
    nominal_boundary_perturbation: float | None = None,
    additive_boundary_condition: str | Mapping[str, Any] | None = None,
) -> PoissonBoundarySensitivity:
    """Evaluate a frozen model under scaled or additive Dirichlet data.

    In scaled mode, the family is ``g_a = a*g``.  In additive mode, it is
    ``g_epsilon = g + epsilon*q``, where ``q`` is another configured harmonic
    boundary condition (constant one by default).  The corresponding exact
    solutions are available in closed form because both harmonic extensions
    have zero Laplacian; the right-hand side remains unchanged.
    """

    mode = str(boundary_mode).strip().lower()
    if mode not in {"scaled", "additive"}:
        raise ValueError("boundary_mode must be 'scaled' or 'additive'")
    perturbations = tuple(float(value) for value in perturbation_values)
    if not perturbations:
        raise ValueError("perturbation_values must contain at least one value")
    if any(not math.isfinite(value) for value in perturbations):
        raise ValueError("Boundary perturbations must be finite")
    if len(set(perturbations)) != len(perturbations):
        raise ValueError("perturbation_values must not contain duplicates")
    nominal = float(
        (1.0 if mode == "scaled" else 0.0)
        if nominal_boundary_perturbation is None
        else nominal_boundary_perturbation
    )
    if not math.isfinite(nominal):
        raise ValueError("nominal_boundary_perturbation must be finite")
    nominal_indices = [
        index
        for index, value in enumerate(perturbations)
        if math.isclose(value, nominal, rel_tol=1e-12, abs_tol=1e-12)
    ]
    if not nominal_indices:
        raise ValueError(
            "perturbation_values must include nominal boundary perturbation"
        )

    resolved_boundary = resolve_poisson_boundary_condition(boundary_condition)
    resolved_additive = None
    if mode == "additive":
        resolved_additive = resolve_poisson_boundary_condition(
            {"name": "constant", "value": 1.0}
            if additive_boundary_condition is None
            else additive_boundary_condition
        )

    parameter = next(
        (value for value in model.parameters() if value.requires_grad),
        None,
    )
    if parameter is None:
        raise ValueError("The model has no trainable parameters")
    evaluation = points.evaluation.to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    boundary = points.boundary.to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = _scalar_prediction(model, evaluation)
            boundary_prediction = _scalar_prediction(model, boundary)
            sine_solution = exact_poisson_solution(evaluation)
            base_extension = poisson_boundary_values(
                evaluation,
                resolved_boundary,
            )
            base_boundary = poisson_boundary_values(
                boundary,
                resolved_boundary,
            )
            additive_extension = (
                poisson_boundary_values(evaluation, resolved_additive)
                if resolved_additive is not None
                else torch.zeros_like(base_extension)
            )
            additive_boundary = (
                poisson_boundary_values(boundary, resolved_additive)
                if resolved_additive is not None
                else torch.zeros_like(base_boundary)
            )

            def boundary_parameters(value: float) -> tuple[float, float]:
                return (value, 0.0) if mode == "scaled" else (1.0, value)

            nominal_scale, nominal_additive = boundary_parameters(nominal)
            nominal_exact = (
                sine_solution
                + nominal_scale * base_extension
                + nominal_additive * additive_extension
            )
            nominal_boundary = (
                nominal_scale * base_boundary
                + nominal_additive * additive_boundary
            )
            nominal_solution_norm = torch.linalg.vector_norm(
                nominal_exact
            ).clamp_min(torch.finfo(nominal_exact.dtype).eps)

            raw_metrics = []
            for perturbation in perturbations:
                scale, additive = boundary_parameters(perturbation)
                exact = (
                    sine_solution
                    + scale * base_extension
                    + additive * additive_extension
                )
                target = scale * base_boundary + additive * additive_boundary
                raw_metrics.append(
                    (
                        float(
                            torch.linalg.vector_norm(prediction - exact)
                            / nominal_solution_norm
                        ),
                        float(
                            (boundary_prediction - target)
                            .square()
                            .mean()
                            .sqrt()
                        ),
                        float(target.square().mean().sqrt()),
                        float(
                            (target - nominal_boundary)
                            .square()
                            .mean()
                            .sqrt()
                        ),
                    )
                )

        pde_residual_rmse = float(
            poisson_residual(
                model,
                evaluation,
                create_graph=False,
            )
            .detach()
            .square()
            .mean()
            .sqrt()
        )
        nominal_solution_error, nominal_boundary_error, _, _ = raw_metrics[
            nominal_indices[0]
        ]
        evaluations = []
        for perturbation, raw in zip(perturbations, raw_metrics):
            solution_error, boundary_error, value_rmse, change_rmse = raw
            scale, additive = boundary_parameters(perturbation)
            evaluations.append(
                PoissonBoundaryEvaluation(
                    boundary_mode=mode,
                    boundary_perturbation=perturbation,
                    boundary_scale=scale,
                    boundary_additive_scale=additive,
                    boundary_value_rmse=value_rmse,
                    boundary_change_rmse_from_nominal=change_rmse,
                    nominal_normalized_l2_error=solution_error,
                    pde_residual_rmse=pde_residual_rmse,
                    boundary_condition_rmse=boundary_error,
                    nominal_normalized_l2_error_change_from_nominal=(
                        solution_error - nominal_solution_error
                    ),
                    boundary_condition_rmse_change_from_nominal=(
                        boundary_error - nominal_boundary_error
                    ),
                )
            )
        evaluation_tuple = tuple(evaluations)
        perturbed = [
            value
            for value in evaluation_tuple
            if not math.isclose(
                value.boundary_perturbation,
                nominal,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]

        def max_slope(attribute: str, nominal_value: float) -> float:
            if not perturbed:
                return 0.0
            return max(
                abs(getattr(value, attribute) - nominal_value)
                / abs(value.boundary_perturbation - nominal)
                for value in perturbed
            )

        solution_errors = [value[0] for value in raw_metrics]
        boundary_errors = [value[1] for value in raw_metrics]
        return PoissonBoundarySensitivity(
            boundary_mode=mode,
            nominal_boundary_perturbation=nominal,
            boundary_condition=resolved_boundary,
            additive_boundary_condition=resolved_additive,
            evaluations=evaluation_tuple,
            mean_nominal_normalized_l2_error=(
                sum(solution_errors) / len(solution_errors)
            ),
            worst_nominal_normalized_l2_error=max(solution_errors),
            mean_boundary_condition_rmse=(
                sum(boundary_errors) / len(boundary_errors)
            ),
            worst_boundary_condition_rmse=max(boundary_errors),
            pde_residual_rmse=pde_residual_rmse,
            max_nominal_normalized_l2_error_change_per_unit_perturbation=(
                max_slope(
                    "nominal_normalized_l2_error",
                    nominal_solution_error,
                )
            ),
            max_boundary_condition_rmse_change_per_unit_perturbation=(
                max_slope(
                    "boundary_condition_rmse",
                    nominal_boundary_error,
                )
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
    nominal_forcing_scale: float | None = None,
    forcing_mode: str = "scaled",
    nominal_forcing_perturbation: float | None = None,
    additive_solution_terms: int = 50,
    boundary_condition: str | Mapping[str, Any] | None = None,
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

    mode = _forcing_mode(forcing_mode)
    resolved_boundary = resolve_poisson_boundary_condition(boundary_condition)
    perturbations = tuple(
        _forcing_perturbation(mode, value) for value in forcing_scales
    )
    if nominal_forcing_scale is not None and nominal_forcing_perturbation is not None:
        raise ValueError(
            "Set only one of nominal_forcing_scale and "
            "nominal_forcing_perturbation"
        )
    configured_nominal = (
        nominal_forcing_perturbation
        if nominal_forcing_perturbation is not None
        else nominal_forcing_scale
    )
    nominal_perturbation = _forcing_perturbation(
        mode,
        (1.0 if mode == "scaled" else 0.0)
        if configured_nominal is None
        else configured_nominal
    )
    nominal_scale, nominal_additive = _forcing_parameters(
        mode,
        nominal_perturbation,
    )
    if not perturbations:
        raise ValueError("forcing_scales must contain at least one value")
    if len(set(perturbations)) != len(perturbations):
        raise ValueError("forcing_scales must not contain duplicates")
    if mode == "scaled" and nominal_perturbation == 0.0:
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
        boundary_target = poisson_boundary_values(boundary, resolved_boundary)
        observation_parts = []
        if pde_weight > 0.0:
            observation_parts.append(
                math.sqrt(pde_weight / residual.numel()) * residual
            )
        if boundary_weight > 0.0:
            observation_parts.append(
                math.sqrt(boundary_weight / boundary_prediction.numel())
                    * (boundary_prediction - boundary_target)
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
        boundary_solution = poisson_boundary_values(
            evaluation,
            resolved_boundary,
        )
        additive_solution = (
            constant_poisson_solution(
                evaluation,
                series_terms=additive_solution_terms,
            )
            if mode == "additive"
            else torch.zeros_like(reference_solution)
        )
        reference_forcing = poisson_forcing(evaluation)
        nominal_solution_rms = (
            nominal_scale * reference_solution
            + nominal_additive * additive_solution
            + boundary_solution
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

        evaluation_boundary_prediction = _scalar_prediction(model, boundary)
        boundary_condition_rmse = metric_summary(
            (evaluation_boundary_prediction - boundary_target)
            .square()
            .mean()
            .clamp_min(tiny)
            .sqrt()
        )

        evaluations = []
        for perturbation in perturbations:
            scale, additive = _forcing_parameters(mode, perturbation)
            exact = (
                scale * reference_solution
                + additive * additive_solution
                + boundary_solution
            )
            difference_mse = (prediction - exact).square().mean()
            nominal_normalized_l2 = (
                difference_mse.clamp_min(tiny).sqrt() / nominal_solution_rms
            )
            pde_residual = operator_prediction - (
                scale * reference_forcing + additive
            )
            residual_rmse = pde_residual.square().mean().clamp_min(tiny).sqrt()
            evaluations.append(
                PoissonLaplaceForcingEvaluation(
                    forcing_mode=mode,
                    forcing_perturbation=perturbation,
                    forcing_scale=scale,
                    nominal_normalized_l2_error=metric_summary(
                        nominal_normalized_l2
                    ),
                    pde_residual_rmse=metric_summary(residual_rmse),
                    boundary_condition_rmse=boundary_condition_rmse,
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
            forcing_mode=mode,
            nominal_forcing_perturbation=nominal_perturbation,
            additive_solution_terms=(
                int(additive_solution_terms) if mode == "additive" else None
            ),
        )
    finally:
        model.train(was_training)
