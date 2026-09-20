"""Function-preserving rescaling for sequential linear parameterizations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .layers import DiagLinear


LinearLayer = nn.Linear | DiagLinear
_SUPPORTED_HOMOGENEOUS_ACTIVATIONS = (nn.Identity, nn.ReLU, nn.LeakyReLU)


@dataclass(frozen=True)
class LinearRescalingResult:
    """The sampled hidden-coordinate scales used for a linear rescaling."""

    mode: str
    log_scale_std: float
    seed: int
    hidden_log_scales: tuple[tuple[float, ...], ...]

    @property
    def hidden_scales(self) -> tuple[tuple[float, ...], ...]:
        """Return the positive hidden-coordinate scales."""

        return tuple(
            tuple(math.exp(value) for value in layer_values)
            for layer_values in self.hidden_log_scales
        )


def _dimensions(layer: LinearLayer) -> tuple[int, int]:
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    return layer.dim, layer.dim


def _resolve_linear_layers(
    model: nn.Module,
    layers: Sequence[LinearLayer] | None,
) -> list[LinearLayer]:
    if layers is not None:
        resolved = list(layers)
    elif hasattr(model, "layers"):
        resolved = list(model.layers)
        activations = list(getattr(model, "activations", ()))
        hidden_activations = activations[:-1] if activations else []
        if any(
            not isinstance(activation, _SUPPORTED_HOMOGENEOUS_ACTIVATIONS)
            for activation in hidden_activations
        ):
            raise ValueError(
                "Function-preserving rescaling requires identity or positively "
                "homogeneous hidden activations (ReLU or LeakyReLU)"
            )
    elif isinstance(model, nn.Sequential):
        resolved = []
        for module in model:
            if isinstance(module, (nn.Linear, DiagLinear)):
                resolved.append(module)
            elif not isinstance(module, _SUPPORTED_HOMOGENEOUS_ACTIVATIONS):
                raise ValueError(
                    "Sequential rescaling supports only linear layers and identity, "
                    "ReLU, or LeakyReLU activations"
                )
    else:
        raise ValueError(
            "Could not infer an ordered linear chain; pass its layers explicitly"
        )

    if len(resolved) < 2:
        raise ValueError("Function-preserving rescaling requires at least two layers")
    if any(not isinstance(layer, (nn.Linear, DiagLinear)) for layer in resolved):
        raise TypeError("Every rescaled layer must be nn.Linear or DiagLinear")

    for left, right in zip(resolved, resolved[1:]):
        _, left_output = _dimensions(left)
        right_input, _ = _dimensions(right)
        if left_output != right_input:
            raise ValueError(
                "Adjacent linear layers have incompatible dimensions: "
                f"{left_output} and {right_input}"
            )
    return resolved


def apply_function_preserving_linear_rescaling(
    model: nn.Module,
    *,
    log_scale_std: float = 1.0,
    seed: int = 0,
    mode: str = "neuronwise",
    layers: Sequence[LinearLayer] | None = None,
) -> LinearRescalingResult:
    """Randomly rescale a sequential linear network without changing its function.

    At every hidden interface, a positive diagonal matrix ``D`` is sampled. For
    consecutive dense layers, the transformation has the form

        W_left  <- D @ W_left
        b_left  <- D @ b_left
        W_right <- W_right @ inverse(D).

    Applying this simultaneously at every interface preserves the complete
    affine function, including biases. Diagonal layers use the corresponding
    elementwise transformation. ``neuronwise`` samples one scale per hidden
    coordinate; ``layerwise`` shares one scale across an entire hidden layer.
    ``coordinatewise`` is accepted as an alias for ``neuronwise``.

    The inferred model must be a chain of ``nn.Linear`` and/or ``DiagLinear``
    layers whose hidden activations are identity, ReLU, or LeakyReLU. Positive
    rescaling commutes with those activations, so the same proof applies to these
    nonlinear networks. For a custom model, pass the ordered layers explicitly
    and ensure that intervening operations have the same positive-homogeneity
    property. Apply rescaling before constructing an optimizer, because existing
    optimizer state is not rescaled.
    """

    standard_deviation = float(log_scale_std)
    if not math.isfinite(standard_deviation) or standard_deviation < 0.0:
        raise ValueError("log_scale_std must be non-negative and finite")

    normalized_mode = str(mode).lower().replace("-", "_")
    if normalized_mode == "coordinatewise":
        normalized_mode = "neuronwise"
    if normalized_mode not in {"layerwise", "neuronwise"}:
        raise ValueError("mode must be 'layerwise' or 'neuronwise'")

    resolved = _resolve_linear_layers(model, layers)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    hidden_log_scales: list[torch.Tensor] = []
    for layer in resolved[:-1]:
        _, width = _dimensions(layer)
        shape = (1,) if normalized_mode == "layerwise" else (width,)
        sampled = torch.randn(
            shape,
            generator=generator,
            device="cpu",
            dtype=torch.float64,
        ) * standard_deviation
        if normalized_mode == "layerwise":
            sampled = sampled.expand(width).clone()
        hidden_log_scales.append(sampled)

    transformed: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    for index, layer in enumerate(resolved):
        input_width, output_width = _dimensions(layer)
        left = (
            hidden_log_scales[index]
            if index < len(hidden_log_scales)
            else torch.zeros(output_width, dtype=torch.float64)
        )
        right = (
            hidden_log_scales[index - 1]
            if index > 0
            else torch.zeros(input_width, dtype=torch.float64)
        )

        if isinstance(layer, nn.Linear):
            multiplier = torch.exp(left[:, None] - right[None, :]).to(
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
        else:
            multiplier = torch.exp(left - right).to(
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
        new_weight = layer.weight.detach() * multiplier

        new_bias = None
        if layer.bias is not None:
            bias_multiplier = torch.exp(left).to(
                device=layer.bias.device,
                dtype=layer.bias.dtype,
            )
            new_bias = layer.bias.detach() * bias_multiplier

        if not torch.isfinite(new_weight).all().item() or (
            new_bias is not None and not torch.isfinite(new_bias).all().item()
        ):
            raise FloatingPointError(
                "The sampled rescaling is not finite in the model's parameter dtype; "
                "reduce log_scale_std"
            )
        transformed.append((new_weight, new_bias))

    with torch.no_grad():
        for layer, (new_weight, new_bias) in zip(resolved, transformed):
            layer.weight.copy_(new_weight)
            if layer.bias is not None:
                assert new_bias is not None
                layer.bias.copy_(new_bias)

    return LinearRescalingResult(
        mode=normalized_mode,
        log_scale_std=standard_deviation,
        seed=int(seed),
        hidden_log_scales=tuple(
            tuple(float(value) for value in values)
            for values in hidden_log_scales
        ),
    )
