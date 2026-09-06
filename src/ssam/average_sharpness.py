from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class AverageSharpnessResult:
    """Result of a Monte Carlo average-sharpness evaluation."""

    clean_loss: float
    regularized_loss: float
    average_sharpness: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    perturbation_samples: int
    sharpness_scale: float


@torch.no_grad()
def evaluate_average_sharpness(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sharpness_scale: float,
    *,
    samples: int = 4096,
    seed: int = 12345,
    antithetic: bool = True,
    confidence_multiplier: float = 1.96,
) -> AverageSharpnessResult:
    """Estimate Gaussian average sharpness at the model's current parameters.

    The estimated quantity is

        E_Z[L(theta + eta*Z)] - L(theta),

    where Z is a standard Gaussian parameter vector and ``eta`` is
    ``sharpness_scale``.

    Every loss evaluation uses the complete dataset represented by
    ``data_loader``. Its batch size only controls evaluation memory usage; it
    does not change the mathematical quantity being estimated.

    For fair comparisons between trained models, use:

    - the same evaluation DataLoader;
    - the same ``sharpness_scale``;
    - the same number of samples;
    - the same seed;
    - models with the same parameter structure.

    Using the same seed gives the models common Gaussian perturbations, which
    usually makes differences between models much more accurate.

    The loss function must return a scalar mean loss for each batch.
    """

    if sharpness_scale < 0.0 or not math.isfinite(sharpness_scale):
        raise ValueError(
            "sharpness_scale must be non-negative and finite"
        )

    if samples < 2:
        raise ValueError("At least two perturbation samples are required")

    if antithetic and samples % 2:
        raise ValueError(
            "Antithetic evaluation requires an even number of samples"
        )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise ValueError("The model has no trainable parameters")

    device = parameters[0].device

    # Save the exact model state so evaluation has no lasting side effects.
    clean_parameters = [
        parameter.detach().clone()
        for parameter in parameters
    ]

    buffers = list(model.buffers())
    clean_buffers = [
        buffer.detach().clone()
        for buffer in buffers
    ]

    # Preserve the training/evaluation state of every submodule.
    module_training_states = [
        (module, module.training)
        for module in model.modules()
    ]

    # Generate noise on the CPU using a private generator. This prevents the
    # evaluation from modifying the application's global PyTorch RNG state.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def restore_clean_state() -> None:
        """Restore the unperturbed parameters and model buffers."""

        for parameter, clean_parameter in zip(
            parameters,
            clean_parameters,
        ):
            parameter.copy_(clean_parameter)

        for buffer, clean_buffer in zip(
            buffers,
            clean_buffers,
        ):
            buffer.copy_(clean_buffer)

    def complete_dataset_loss() -> float:
        """Evaluate the mean loss over every example in the DataLoader."""

        total_loss = 0.0
        total_examples = 0

        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_loss = loss_fn(model(inputs), targets)

            if batch_loss.numel() != 1:
                raise ValueError(
                    "loss_fn must return one scalar mean loss per batch"
                )

            if not torch.isfinite(batch_loss).item():
                raise FloatingPointError(
                    "A non-finite loss was encountered during sharpness "
                    "evaluation"
                )

            batch_size = int(targets.shape[0])

            # loss_fn is assumed to return the mean for this batch. Multiplying
            # by batch_size allows a correctly weighted complete-dataset mean,
            # including when the final batch is smaller.
            total_loss += float(batch_loss.item()) * batch_size
            total_examples += batch_size

        if total_examples == 0:
            raise ValueError("The evaluation DataLoader is empty")

        return total_loss / total_examples

    def draw_gaussian_noise() -> list[torch.Tensor]:
        """Draw one standard Gaussian vector matching all parameters."""

        noise = []

        for parameter in parameters:
            # Draw in float64 for a reproducible, high-quality random sequence,
            # then convert to the parameter's device and dtype.
            sample = torch.randn(
                parameter.shape,
                generator=generator,
                device="cpu",
                dtype=torch.float64,
            )

            noise.append(
                sample.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )

        return noise

    def evaluate_perturbation(
        noise: list[torch.Tensor],
        sign: float,
    ) -> float:
        """Evaluate L(theta + sign*eta*Z)."""

        for parameter, clean_parameter, sample in zip(
            parameters,
            clean_parameters,
            noise,
        ):
            parameter.copy_(
                clean_parameter
                + sign * sharpness_scale * sample
            )

        # Each evaluation starts with the same buffers. This matters for models
        # containing stateful components such as BatchNorm.
        for buffer, clean_buffer in zip(
            buffers,
            clean_buffers,
        ):
            buffer.copy_(clean_buffer)

        return complete_dataset_loss()

    model.eval()

    try:
        restore_clean_state()
        clean_loss = complete_dataset_loss()

        # Welford's algorithm accumulates the mean and variance accurately
        # without storing thousands of loss values.
        observation_count = 0
        mean_sharpness = 0.0
        squared_difference_sum = 0.0

        if antithetic:
            # Treat each (+Z, -Z) pair as one independent observation. Computing
            # the standard error from pair averages correctly accounts for the
            # correlation between the two members of each pair.
            pair_count = samples // 2

            for _ in range(pair_count):
                noise = draw_gaussian_noise()

                positive_loss = evaluate_perturbation(noise, +1.0)
                negative_loss = evaluate_perturbation(noise, -1.0)

                pair_sharpness = (
                    0.5 * (positive_loss + negative_loss)
                    - clean_loss
                )

                observation_count += 1
                difference = pair_sharpness - mean_sharpness
                mean_sharpness += difference / observation_count
                squared_difference_sum += (
                    difference
                    * (pair_sharpness - mean_sharpness)
                )
        else:
            for _ in range(samples):
                noise = draw_gaussian_noise()
                perturbed_loss = evaluate_perturbation(noise, +1.0)
                sample_sharpness = perturbed_loss - clean_loss

                observation_count += 1
                difference = sample_sharpness - mean_sharpness
                mean_sharpness += difference / observation_count
                squared_difference_sum += (
                    difference
                    * (sample_sharpness - mean_sharpness)
                )

        sample_variance = (
            squared_difference_sum / (observation_count - 1)
        )
        standard_error = math.sqrt(
            sample_variance / observation_count
        )

        confidence_radius = (
            confidence_multiplier * standard_error
        )

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
        # The trained parameter estimate and all buffers are restored even when
        # evaluation raises an exception.
        restore_clean_state()

        for module, was_training in module_training_states:
            module.training = was_training
