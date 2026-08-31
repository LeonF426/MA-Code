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

    def step(
        self,
        loss_closure: LossClosure,
        sharpness_scale: float,
    ) -> float:
        if self.samples < 1:
            raise ValueError("S-SAM requires at least one perturbation sample")

        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        # Preserve the clean parameters θ.
        clean_parameters = [
            parameter.detach().clone()
            for parameter in parameters
        ]

        with torch.no_grad():
            clean_loss = loss_closure()

        if not torch.isfinite(clean_loss).item():
            raise FloatingPointError(
                "The clean loss is already non-finite before applying "
                f"an S-SAM perturbation: clean_loss={clean_loss.item()}"
            )

        # These tensors hold the running mean of the sampled gradients.
        mean_gradients = [
            torch.zeros_like(parameter)
            for parameter in parameters
        ]

        mean_loss = 0.0

        try:
            for sample_index in range(self.samples):
                perturbations = self._noise(
                    parameters,
                    sharpness_scale,
                )

                # Evaluate this sample at θ + ξ_s.
                with torch.no_grad():
                    for parameter, clean, perturbation in zip(
                        parameters,
                        clean_parameters,
                        perturbations,
                    ):
                        parameter.copy_(clean + perturbation)

                self.optimizer.zero_grad(set_to_none=True)

                loss = loss_closure()

                if loss.numel() != 1:
                    raise ValueError(
                        "The S-SAM loss closure must return a scalar loss"
                    )

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        "Non-finite perturbed loss encountered "
                        f"at sample {sample_index + 1}/{self.samples}, "
                        f"sharpness_scale={sharpness_scale}"
                    )

                loss.backward()

                # Online mean weight:
                # mean_s = mean_{s-1} + (value_s - mean_{s-1}) / s
                mean_weight = 1.0 / (sample_index + 1)

                for mean_gradient, parameter in zip(
                    mean_gradients,
                    parameters,
                ):
                    gradient = parameter.grad

                    if gradient is None:
                        # Treat a missing gradient as zero for this sample.
                        mean_gradient.mul_(1.0 - mean_weight)
                        continue

                    if not torch.isfinite(gradient).all().item():
                        raise FloatingPointError(
                            "Non-finite perturbed gradient encountered "
                            f"at sample {sample_index + 1}/{self.samples}, "
                            f"sharpness_scale={sharpness_scale}"
                        )

                    # Numerically safer than summing all gradients first.
                    mean_gradient.lerp_(
                        gradient.detach(),
                        mean_weight,
                    )

                loss_value = float(loss.detach().item())
                mean_loss += (
                    loss_value - mean_loss
                ) * mean_weight

        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise

        finally:
            # Always return the model to the clean parameters θ,
            # including when a sampled loss fails.
            with torch.no_grad():
                for parameter, clean in zip(
                    parameters,
                    clean_parameters,
                ):
                    parameter.copy_(clean)

        # Apply the averaged perturbed gradient to the clean parameters.
        self.optimizer.zero_grad(set_to_none=True)

        for parameter, mean_gradient in zip(
            parameters,
            mean_gradients,
        ):
            parameter.grad = mean_gradient

        self.optimizer.step()

        return mean_loss

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
