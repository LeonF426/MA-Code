"""Configuration-driven model construction."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from .config import model_config
from .layers import DiagLinear
from .rescaling import (
    apply_function_preserving_linear_rescaling,
    apply_linear_interface_rescaling,
)


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


def register_activation(name: str, factory: Callable[[], nn.Module]) -> None:
    """Register an activation for use in model dictionaries."""

    _ACTIVATIONS[name.lower()] = factory


def activation_from_name(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown activation {name!r}. Available: {sorted(_ACTIVATIONS)}"
        ) from exc


class ConfigurableNet(nn.Module):
    """A dense/diagonal feed-forward network assembled from layer dictionaries."""

    def __init__(
        self,
        layer_specs: list[Mapping[str, Any]],
        default_activation: str = "identity",
        output_activation: str = "identity",
        default_bias: bool = True,
        output_reduction: str = "none",
    ) -> None:
        super().__init__()
        if not layer_specs:
            raise ValueError("At least one layer is required")

        layers: list[nn.Module] = []
        activations: list[nn.Module] = []
        for index, spec in enumerate(layer_specs):
            layer_type = str(spec.get("type", "dense")).lower()
            in_dim = int(spec["in_dim"])
            out_dim = int(spec.get("out_dim", in_dim))
            bias = bool(spec.get("bias", default_bias))
            if layer_type == "diag":
                if in_dim != out_dim:
                    raise ValueError("Diagonal layers require in_dim == out_dim")
                layer = DiagLinear(in_dim, bias=bias)
            elif layer_type == "dense":
                layer = nn.Linear(in_dim, out_dim, bias=bias)
            else:
                raise ValueError(f"Unknown layer type {layer_type!r}")
            layers.append(layer)
            activation = (
                output_activation
                if index == len(layer_specs) - 1
                else str(spec.get("activation", default_activation))
            )
            activations.append(activation_from_name(activation))

        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)
        self.output_reduction = output_reduction.lower()
        if self.output_reduction not in {"none", "sum", "mean"}:
            raise ValueError("output_reduction must be 'none', 'sum', or 'mean'")

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs
        for layer, activation in zip(self.layers, self.activations):
            value = activation(layer(value))
        return value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward_features(inputs)
        if self.output_reduction == "sum":
            return outputs.sum(dim=-1)
        if self.output_reduction == "mean":
            return outputs.mean(dim=-1)
        return outputs


MixedLinearNet = ConfigurableNet


def _expanded_layer_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "layers" in config:
        specs = [dict(spec) for spec in config["layers"]]
        previous = int(config["input_dim"])
        for spec in specs:
            spec.setdefault("in_dim", previous)
            spec.setdefault("out_dim", spec["in_dim"])
            previous = int(spec["out_dim"])
        return specs

    depth = int(config["depth"])
    input_dim = int(config["input_dim"])
    output_dim = int(config.get("output_dim", 1))
    width = config.get("width", 64)
    hidden_widths = [int(width)] * (depth - 1) if isinstance(width, int) else list(width)
    if len(hidden_widths) != depth - 1:
        raise ValueError("A width list must contain depth - 1 entries")
    dimensions = [input_dim, *hidden_widths, output_dim]
    return [
        {"type": "dense", "in_dim": source, "out_dim": target}
        for source, target in zip(dimensions, dimensions[1:])
    ]


def _initialize_tensor(tensor: torch.Tensor, name: str, options: Mapping[str, Any]) -> None:
    if name == "default":
        return
    if name == "zeros":
        nn.init.zeros_(tensor)
    elif name == "ones":
        nn.init.ones_(tensor)
    elif name == "normal":
        nn.init.normal_(tensor, mean=float(options.get("mean", 0.0)), std=float(options.get("std", 0.02)))
    elif name == "uniform":
        nn.init.uniform_(tensor, a=float(options.get("low", -0.1)), b=float(options.get("high", 0.1)))
    elif name == "identity":
        if tensor.ndim == 1:
            nn.init.ones_(tensor)
        elif tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1]:
            nn.init.eye_(tensor)
        else:
            raise ValueError("Identity initialization requires a square or diagonal layer")
    elif tensor.ndim < 2:
        bound = math.sqrt(3.0 / max(1, tensor.numel()))
        nn.init.uniform_(tensor, -bound, bound)
    elif name == "xavier_uniform":
        nn.init.xavier_uniform_(tensor, gain=float(options.get("gain", 1.0)))
    elif name == "xavier_normal":
        nn.init.xavier_normal_(tensor, gain=float(options.get("gain", 1.0)))
    elif name == "kaiming_uniform":
        nn.init.kaiming_uniform_(tensor, nonlinearity=str(options.get("nonlinearity", "relu")))
    elif name == "kaiming_normal":
        nn.init.kaiming_normal_(tensor, nonlinearity=str(options.get("nonlinearity", "relu")))
    elif name == "orthogonal":
        nn.init.orthogonal_(tensor, gain=float(options.get("gain", 1.0)))
    else:
        raise ValueError(f"Unknown parameter initialization {name!r}")


def initialize_model(model: nn.Module, initialization: str | Mapping[str, Any]) -> None:
    options = {"name": initialization} if isinstance(initialization, str) else dict(initialization)
    init_type = options.pop("type", None)
    alias = options.pop("name", None)
    if init_type is None:
        init_type = "default" if alias is None else alias
    elif alias is not None and str(alias).lower() != str(init_type).lower():
        raise ValueError("parameter_init.type and parameter_init.name disagree")
    name = str(init_type).lower()
    seed = options.pop("seed", None)
    options.pop("rescaling", None)
    allowed_options = {
        "default": {"bias"},
        "pretrained": set(),
        "zeros": {"bias"},
        "ones": {"bias"},
        "identity": {"bias"},
        "normal": {"mean", "std", "bias"},
        "uniform": {"low", "high", "bias"},
        "xavier_uniform": {"gain", "bias"},
        "xavier_normal": {"gain", "bias"},
        "kaiming_uniform": {"nonlinearity", "bias"},
        "kaiming_normal": {"nonlinearity", "bias"},
        "orthogonal": {"gain", "bias"},
    }
    if name not in allowed_options:
        raise ValueError(f"Unknown parameter initialization {name!r}")
    unknown = set(options) - allowed_options[name]
    if unknown:
        raise ValueError(f"Unknown parameter_init keys for {name!r}: {sorted(unknown)}")
    if name == "pretrained":
        return
    rng_context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with rng_context:
        if seed is not None:
            torch.random.default_generator.manual_seed(int(seed))
        for module in model.modules():
            if isinstance(module, (nn.Linear, DiagLinear)):
                _initialize_tensor(module.weight, name, options)
                if module.bias is not None:
                    nn.init.constant_(module.bias, float(options.get("bias", 0.0)))


def _build_torchvision_model(config: Mapping[str, Any]) -> nn.Module:
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "Torchvision benchmarks require `pip install -e '.[benchmarks]'`."
        ) from exc

    model_name = str(config["type"]).split("/", 1)[1]
    if not hasattr(models, model_name):
        raise ValueError(f"Unknown torchvision model {model_name!r}")
    init_config = config.get("parameter_init", {"type": "default"})
    init_name = (
        init_config
        if isinstance(init_config, str)
        else init_config.get("type", init_config.get("name", "default"))
    )
    pretrained = str(init_name).lower() == "pretrained"
    model = getattr(models, model_name)(weights="DEFAULT" if pretrained else None)
    num_classes = int(config.get("num_classes", 1000))
    if num_classes != 1000:
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"):
            classifier = model.classifier
            if isinstance(classifier, nn.Linear):
                model.classifier = nn.Linear(classifier.in_features, num_classes)
            elif isinstance(classifier, nn.Sequential):
                last = next(i for i in range(len(classifier) - 1, -1, -1) if isinstance(classifier[i], nn.Linear))
                classifier[last] = nn.Linear(classifier[last].in_features, num_classes)
            else:
                raise ValueError(f"Cannot replace classifier for {model_name!r}")
        elif hasattr(model, "heads") and hasattr(model.heads, "head"):
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        else:
            raise ValueError(f"Cannot replace classifier for {model_name!r}")
    # Also call this for pretrained models: the initializer returns without
    # changing weights, but still validates misspelled/unsupported options.
    initialize_model(model, init_config)
    return model


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build a configured custom or torchvision benchmark model."""

    resolved = model_config(config)
    initialization = resolved.get("parameter_init", {"type": "default"})
    initialization_options = (
        {} if isinstance(initialization, str) else dict(initialization)
    )
    seed = initialization_options.get("seed")
    rng_context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with rng_context:
        if seed is not None:
            torch.random.default_generator.manual_seed(int(seed))
        if resolved["type"].startswith("torchvision/"):
            if (
                isinstance(initialization, Mapping)
                and initialization.get("rescaling")
            ):
                raise ValueError(
                    "parameter_init.rescaling is not supported for torchvision models"
                )
            return _build_torchvision_model(resolved)

        model = ConfigurableNet(
            _expanded_layer_specs(resolved),
            default_activation=str(resolved.get("activation", "identity")),
            output_activation=str(resolved.get("output_activation", "identity")),
            default_bias=bool(resolved.get("bias", True)),
            output_reduction=str(resolved.get("output_reduction", "none")),
        )
        initialize_model(model, initialization)
    if isinstance(initialization, Mapping):
        rescaling = initialization.get("rescaling")
        if rescaling:
            options = {} if rescaling is True else dict(rescaling)
            if bool(options.pop("enabled", True)):
                preserve_function = bool(options.pop("preserve_function", True))
                rescaling_function = (
                    apply_function_preserving_linear_rescaling
                    if preserve_function
                    else apply_linear_interface_rescaling
                )
                model.rescaling_result = rescaling_function(
                    model,
                    log_scale_std=float(options.pop("log_scale_std", 1.0)),
                    seed=int(options.pop("seed", 0)),
                    mode=str(options.pop("mode", "neuronwise")),
                )
                if options:
                    raise ValueError(
                        "Unknown parameter_init.rescaling keys: "
                        f"{sorted(options)}"
                    )
    return model

