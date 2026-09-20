"""Monte Carlo average-sharpness diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class AverageSharpnessResult:
    """A Gaussian average-sharpness estimate and its Monte Carlo uncertainty."""

    clean_loss: float
    regularized_loss: float
    average_sharpness: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    perturbation_samples: int
    sharpness_scale: float


@dataclass(frozen=True)
class AverageSharpnessInterpolationPoint:
    """Loss and average sharpness at one point between two parameter vectors."""

    coefficient: float
    clean_loss: float
    regularized_loss: float
    average_sharpness: float
    standard_error: float
    confidence_low: float
    confidence_high: float


@dataclass(frozen=True)
class AverageSharpnessInterpolationResult:
    """Average-sharpness measurements along a straight parameter-space path."""

    points: tuple[AverageSharpnessInterpolationPoint, ...]
    perturbation_samples: int
    sharpness_scale: float
    seed: int
    antithetic: bool
    normalized: bool
    confidence_multiplier: float


def evaluate_average_sharpness_closure(
    model: nn.Module,
    loss_closure: Callable[[], torch.Tensor],
    sharpness_scale: float,
    *,
    samples: int = 4096,
    seed: int = 12345,
    antithetic: bool = True,
    normalized: bool = False,
    requires_grad: bool = False,
    confidence_multiplier: float = 1.96,
) -> AverageSharpnessResult:
    """Estimate ``E[L(theta + eta Z)] - L(theta)`` from a loss closure.

    ``requires_grad`` enables autograd only while evaluating the closure. This is
    needed when the loss itself contains coordinate derivatives, while its False
    default preserves the inexpensive supervised evaluation path. A private CPU
    generator makes an equal seed produce common perturbations for equal model
    architectures without changing the application's global RNG state.
    """

    if sharpness_scale < 0.0 or not math.isfinite(sharpness_scale):
        raise ValueError("sharpness_scale must be non-negative and finite")
    if samples < 2:
        raise ValueError("At least two perturbation samples are required")
    if antithetic and samples % 2:
        raise ValueError("Antithetic evaluation requires an even number of samples")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters")

    clean_parameters = [parameter.detach().clone() for parameter in parameters]
    buffers = list(model.buffers())
    clean_buffers = [buffer.detach().clone() for buffer in buffers]
    module_training_states = [(module, module.training) for module in model.modules()]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def restore_clean_state() -> None:
        with torch.no_grad():
            for parameter, clean in zip(parameters, clean_parameters):
                parameter.copy_(clean)
            for buffer, clean in zip(buffers, clean_buffers):
                buffer.copy_(clean)

    def evaluate_loss() -> float:
        with torch.set_grad_enabled(requires_grad):
            loss = loss_closure()
        if loss.numel() != 1:
            raise ValueError("loss_closure must return one scalar loss")
        if not torch.isfinite(loss).item():
            raise FloatingPointError("A non-finite sharpness loss was encountered")
        return float(loss.detach().item())

    def draw_noise() -> list[torch.Tensor]:
        noise = []
        squared_norm = torch.zeros((), dtype=torch.float64)
        for parameter in parameters:
            sample = torch.randn(
                parameter.shape,
                generator=generator,
                device="cpu",
                dtype=torch.float64,
            )
            noise.append(sample)
            squared_norm += sample.square().sum()
        factor = (
            sharpness_scale / float(torch.sqrt(squared_norm).clamp_min(1e-12))
            if normalized
            else sharpness_scale
        )
        return [
            sample.to(device=parameter.device, dtype=parameter.dtype) * factor
            for parameter, sample in zip(parameters, noise)
        ]

    def evaluate_perturbation(noise: list[torch.Tensor], sign: float) -> float:
        with torch.no_grad():
            for parameter, clean, sample in zip(parameters, clean_parameters, noise):
                parameter.copy_(clean + sign * sample)
            for buffer, clean in zip(buffers, clean_buffers):
                buffer.copy_(clean)
        return evaluate_loss()

    model.eval()
    try:
        restore_clean_state()
        clean_loss = evaluate_loss()
        observation_count = 0
        mean_sharpness = 0.0
        squared_difference_sum = 0.0

        iterations = samples // 2 if antithetic else samples
        for _ in range(iterations):
            noise = draw_noise()
            positive = evaluate_perturbation(noise, +1.0)
            if antithetic:
                negative = evaluate_perturbation(noise, -1.0)
                observation = 0.5 * (positive + negative) - clean_loss
            else:
                observation = positive - clean_loss
            observation_count += 1
            difference = observation - mean_sharpness
            mean_sharpness += difference / observation_count
            squared_difference_sum += difference * (observation - mean_sharpness)

        # One antithetic pair is a useful point estimate but cannot estimate
        # sampling variance. Report NaN uncertainty instead of dividing by zero.
        if observation_count == 1:
            standard_error = float("nan")
        else:
            sample_variance = squared_difference_sum / (observation_count - 1)
            standard_error = math.sqrt(max(0.0, sample_variance) / observation_count)
        confidence_radius = confidence_multiplier * standard_error
        return AverageSharpnessResult(
            clean_loss=clean_loss,
            regularized_loss=clean_loss + mean_sharpness,
            average_sharpness=mean_sharpness,
            standard_error=standard_error,
            confidence_low=mean_sharpness - confidence_radius,
            confidence_high=mean_sharpness + confidence_radius,
            perturbation_samples=samples,
            sharpness_scale=sharpness_scale,
        )
    finally:
        restore_clean_state()
        for module, was_training in module_training_states:
            module.training = was_training


def evaluate_average_sharpness_interpolation_closure(
    model: nn.Module,
    endpoint_model: nn.Module,
    loss_closure: Callable[[], torch.Tensor],
    sharpness_scale: float,
    *,
    interpolation_points: int | Sequence[float] = 21,
    samples: int = 4096,
    seed: int = 12345,
    antithetic: bool = True,
    normalized: bool = False,
    requires_grad: bool = False,
    confidence_multiplier: float = 1.96,
) -> AverageSharpnessInterpolationResult:
    """Measure loss and Gaussian sharpness between two trained models.

    The parameters of ``model`` define coefficient zero and those of
    ``endpoint_model`` define coefficient one. ``loss_closure`` must evaluate
    ``model``; that model is temporarily moved along the straight path

        theta(t) = (1 - t) * theta(0) + t * theta(1).

    The same seed is deliberately reused at every coefficient, producing common
    Gaussian directions and therefore a lower-noise comparison along the path.
    Both models are restored unchanged, including when evaluation raises.
    Buffers are held at their values in ``model`` rather than interpolated.
    """

    if isinstance(interpolation_points, int):
        if interpolation_points < 2:
            raise ValueError("interpolation_points must be at least two")
        coefficients = tuple(
            index / (interpolation_points - 1)
            for index in range(interpolation_points)
        )
    else:
        coefficients = tuple(float(value) for value in interpolation_points)
        if not coefficients:
            raise ValueError("interpolation_points must not be empty")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in coefficients):
            raise ValueError("interpolation coefficients must be finite and in [0, 1]")

    start_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    end_named = [
        (name, parameter)
        for name, parameter in endpoint_model.named_parameters()
        if parameter.requires_grad
    ]
    if not start_named:
        raise ValueError("The model has no trainable parameters")
    if [name for name, _ in start_named] != [name for name, _ in end_named]:
        raise ValueError("The endpoint models must have identical trainable parameter names")
    for (name, start), (_, end) in zip(start_named, end_named):
        if start.shape != end.shape:
            raise ValueError(
                f"Endpoint parameter {name!r} has shapes {tuple(start.shape)} and "
                f"{tuple(end.shape)}"
            )

    parameters = [parameter for _, parameter in start_named]
    start_parameters = [parameter.detach().clone() for parameter in parameters]
    end_parameters = [
        parameter.detach().to(device=start.device, dtype=start.dtype).clone()
        for start, (_, parameter) in zip(parameters, end_named)
    ]
    interpolation_results: list[AverageSharpnessInterpolationPoint] = []

    try:
        for coefficient in coefficients:
            with torch.no_grad():
                for parameter, start, end in zip(
                    parameters,
                    start_parameters,
                    end_parameters,
                ):
                    parameter.copy_(start + coefficient * (end - start))

            measurement = evaluate_average_sharpness_closure(
                model,
                loss_closure,
                sharpness_scale,
                samples=samples,
                seed=seed,
                antithetic=antithetic,
                normalized=normalized,
                requires_grad=requires_grad,
                confidence_multiplier=confidence_multiplier,
            )
            interpolation_results.append(
                AverageSharpnessInterpolationPoint(
                    coefficient=coefficient,
                    clean_loss=measurement.clean_loss,
                    regularized_loss=measurement.regularized_loss,
                    average_sharpness=measurement.average_sharpness,
                    standard_error=measurement.standard_error,
                    confidence_low=measurement.confidence_low,
                    confidence_high=measurement.confidence_high,
                )
            )
    finally:
        with torch.no_grad():
            for parameter, start in zip(parameters, start_parameters):
                parameter.copy_(start)

    return AverageSharpnessInterpolationResult(
        points=tuple(interpolation_results),
        perturbation_samples=samples,
        sharpness_scale=sharpness_scale,
        seed=seed,
        antithetic=antithetic,
        normalized=normalized,
        confidence_multiplier=confidence_multiplier,
    )


def evaluate_average_sharpness(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sharpness_scale: float,
    *,
    samples: int = 4096,
    seed: int = 12345,
    antithetic: bool = True,
    normalized: bool = False,
    requires_grad: bool = False,
    confidence_multiplier: float = 1.96,
) -> AverageSharpnessResult:
    """Estimate average sharpness of a complete-dataset supervised loss.

    Existing callers retain the no-coordinate-gradient, unnormalized Gaussian
    defaults. Set ``requires_grad=True`` for derivative-based loss functions.
    """

    parameter = next((value for value in model.parameters() if value.requires_grad), None)
    if parameter is None:
        raise ValueError("The model has no trainable parameters")
    device = parameter.device

    def complete_dataset_loss() -> torch.Tensor:
        weighted_loss: torch.Tensor | None = None
        total_examples = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_loss = loss_fn(model(inputs), targets)
            if batch_loss.numel() != 1:
                raise ValueError("loss_fn must return one scalar mean loss per batch")
            batch_size = int(targets.shape[0])
            contribution = batch_loss * batch_size
            weighted_loss = contribution if weighted_loss is None else weighted_loss + contribution
            total_examples += batch_size
        if weighted_loss is None or total_examples == 0:
            raise ValueError("The evaluation DataLoader is empty")
        return weighted_loss / total_examples

    return evaluate_average_sharpness_closure(
        model,
        complete_dataset_loss,
        sharpness_scale,
        samples=samples,
        seed=seed,
        antithetic=antithetic,
        normalized=normalized,
        requires_grad=requires_grad,
        confidence_multiplier=confidence_multiplier,
    )


def evaluate_average_sharpness_interpolation(
    model: nn.Module,
    endpoint_model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sharpness_scale: float,
    *,
    interpolation_points: int | Sequence[float] = 21,
    samples: int = 4096,
    seed: int = 12345,
    antithetic: bool = True,
    normalized: bool = False,
    confidence_multiplier: float = 1.96,
) -> AverageSharpnessInterpolationResult:
    """Supervised-data wrapper for the parameter interpolation diagnostic."""

    parameter = next((value for value in model.parameters() if value.requires_grad), None)
    if parameter is None:
        raise ValueError("The model has no trainable parameters")
    device = parameter.device

    def complete_dataset_loss() -> torch.Tensor:
        weighted_loss: torch.Tensor | None = None
        total_examples = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_loss = loss_fn(model(inputs), targets)
            if batch_loss.numel() != 1:
                raise ValueError("loss_fn must return one scalar mean loss per batch")
            batch_size = int(targets.shape[0])
            contribution = batch_loss * batch_size
            weighted_loss = contribution if weighted_loss is None else weighted_loss + contribution
            total_examples += batch_size
        if weighted_loss is None or total_examples == 0:
            raise ValueError("The evaluation DataLoader is empty")
        return weighted_loss / total_examples

    return evaluate_average_sharpness_interpolation_closure(
        model,
        endpoint_model,
        complete_dataset_loss,
        sharpness_scale,
        interpolation_points=interpolation_points,
        samples=samples,
        seed=seed,
        antithetic=antithetic,
        normalized=normalized,
        confidence_multiplier=confidence_multiplier,
    )
