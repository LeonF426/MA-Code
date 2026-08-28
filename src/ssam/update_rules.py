"""Extensible parameter-update rules, including stochastic sharpness-aware SGD."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch
from torch import nn


LossClosure = Callable[[], torch.Tensor]


class UpdateRule(Protocol):
    optimizer: torch.optim.Optimizer

    def step(self, loss_closure: LossClosure, sharpness_scale: float) -> float: ...


def _build_optimizer(parameters, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("name", "sgd")).lower()
    common = {
        "lr": 1.0,
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            momentum=float(config.get("momentum", 0.0)),
            nesterov=bool(config.get("nesterov", False)),
            **common,
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            betas=tuple(config.get("betas", (0.9, 0.999))),
            **common,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            betas=tuple(config.get("betas", (0.9, 0.999))),
            **common,
        )
    raise ValueError(f"Unknown optimizer {name!r}")


class GradientUpdate:
    """A standard optimizer update; data batching determines GD versus SGD."""

    def __init__(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        self.optimizer = _build_optimizer(model.parameters(), config.get("optimizer", {}))

    def step(self, loss_closure: LossClosure, sharpness_scale: float = 0.0) -> float:
        del sharpness_scale
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_closure()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())


class StochasticSharpnessUpdate:
    """S-SAM: update clean parameters using gradients at random perturbations.

    With ``normalized=False`` (the default), every parameter coordinate receives
    Gaussian noise with standard deviation ``sharpness_scale``. With
    ``normalized=True``, one Gaussian direction is normalized to that global
    radius. Multiple samples average independent perturbed gradients.
    """

    def __init__(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        self.model = model
        self.optimizer = _build_optimizer(model.parameters(), config.get("optimizer", {}))
        perturbation = config.get("perturbation", {})
        self.samples = int(perturbation.get("samples", 1))
        self.normalized = bool(perturbation.get("normalized", False))
        distribution = str(perturbation.get("distribution", "gaussian")).lower()
        if distribution != "gaussian":
            raise ValueError("S-SAM currently supports Gaussian perturbations")

    def _noise(self, parameters: list[nn.Parameter], scale: float) -> list[torch.Tensor]:
        noise = [torch.randn_like(parameter) for parameter in parameters]
        if self.normalized:
            norm = torch.sqrt(sum(torch.sum(value.square()) for value in noise)).clamp_min(1e-12)
            return [value * (scale / norm) for value in noise]
        return [value * scale for value in noise]

    def step(self, loss_closure: LossClosure, sharpness_scale: float) -> float:
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        clean = [parameter.detach().clone() for parameter in parameters]
        gradients = [torch.zeros_like(parameter) for parameter in parameters]
        losses: list[float] = []

        try:
            for _ in range(self.samples):
                perturbations = self._noise(parameters, sharpness_scale)
                with torch.no_grad():
                    for parameter, center, perturbation in zip(parameters, clean, perturbations):
                        parameter.copy_(center + perturbation)
                self.optimizer.zero_grad(set_to_none=True)
                loss = loss_closure()
                loss.backward()
                losses.append(float(loss.detach()))
                for total, parameter in zip(gradients, parameters):
                    if parameter.grad is not None:
                        total.add_(parameter.grad)
        finally:
            with torch.no_grad():
                for parameter, center in zip(parameters, clean):
                    parameter.copy_(center)

        self.optimizer.zero_grad(set_to_none=True)
        for parameter, total in zip(parameters, gradients):
            parameter.grad = total.div(self.samples)
        self.optimizer.step()
        return sum(losses) / len(losses)


UpdateFactory = Callable[[nn.Module, Mapping[str, Any]], UpdateRule]
_UPDATE_RULES: dict[str, UpdateFactory] = {
    "gd": GradientUpdate,
    "sgd": GradientUpdate,
    "s_sam": StochasticSharpnessUpdate,
}


def register_update_rule(name: str, factory: UpdateFactory) -> None:
    """Add a custom algorithm without changing the training loop."""

    _UPDATE_RULES[name.lower().replace("-", "_")] = factory


def build_update_rule(model: nn.Module, config: Mapping[str, Any]) -> UpdateRule:
    name = str(config["algorithm"]).lower().replace("-", "_")
    try:
        return _UPDATE_RULES[name](model, config)
    except KeyError as exc:
        raise ValueError(f"Unknown algorithm {name!r}. Available: {sorted(_UPDATE_RULES)}") from exc
