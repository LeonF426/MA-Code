"""Monte Carlo objectives for randomly perturbed model parameters."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


LossClosure = Callable[[], torch.Tensor]


def sample_parameter_noise(
    parameters: list[nn.Parameter],
    scale: float,
    *,
    normalized: bool = False,
) -> list[torch.Tensor]:
    """Draw one Gaussian parameter perturbation.

    ``normalized=False`` gives every coordinate standard deviation ``scale``.
    ``normalized=True`` gives one joint direction with Euclidean norm ``scale``.
    """

    if scale < 0.0:
        raise ValueError("sharpness_scale must be non-negative")
    noise = [torch.randn_like(parameter) for parameter in parameters]
    if normalized:
        squared_norm = sum(value.square().sum() for value in noise)
        norm = torch.sqrt(squared_norm).clamp_min(1e-12)
        return [value * (scale / norm) for value in noise]
    return [value * scale for value in noise]


def estimate_regularized_objective(
    model: nn.Module,
    loss_closure: LossClosure,
    sharpness_scale: float,
    *,
    samples: int = 8,
    normalized: bool = False,
    antithetic: bool = False,
) -> float:
    """Estimate ``E[L(theta + xi)]`` without storing all sampled losses.

    This loss-only helper is intended for diagnostics. S-SAM computes the same
    online average inside its gradient-sampling loop, so training does not need a
    second Monte Carlo pass.
    """

    if samples < 1:
        raise ValueError("samples must be positive")
    if antithetic and samples % 2:
        raise ValueError("antithetic sampling requires an even sample count")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    clean_parameters = [parameter.detach().clone() for parameter in parameters]
    buffers = list(model.buffers())
    clean_buffers = [buffer.detach().clone() for buffer in buffers]
    mean_loss = 0.0
    noise: list[torch.Tensor] | None = None

    try:
        with torch.no_grad():
            for sample_index in range(samples):
                if not antithetic or sample_index % 2 == 0:
                    noise = sample_parameter_noise(
                        parameters, sharpness_scale, normalized=normalized
                    )
                    sign = 1.0
                else:
                    sign = -1.0

                assert noise is not None
                for parameter, clean, perturbation in zip(
                    parameters, clean_parameters, noise
                ):
                    parameter.copy_(clean + sign * perturbation)
                for buffer, clean in zip(buffers, clean_buffers):
                    buffer.copy_(clean)

                loss = loss_closure()
                if loss.numel() != 1:
                    raise ValueError("The loss closure must return a scalar loss")
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        "Non-finite regularized-objective sample encountered at "
                        f"sample {sample_index + 1}/{samples}"
                    )
                weight = 1.0 / (sample_index + 1)
                mean_loss += (float(loss.item()) - mean_loss) * weight
    finally:
        with torch.no_grad():
            for parameter, clean in zip(parameters, clean_parameters):
                parameter.copy_(clean)
            for buffer, clean in zip(buffers, clean_buffers):
                buffer.copy_(clean)

    return mean_loss

