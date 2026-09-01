"""Extensible parameter-update rules, including stochastic sharpness-aware SGD."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from .objectives import sample_parameter_noise
from .schedules import LearningRatePolicy


LossClosure = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class UpdateResult:
    """Values produced by one parameter update."""

    loss: float
    learning_rate: float
    clean_loss: float
    regularized_loss: float | None = None


class UpdateRule(Protocol):
    optimizer: torch.optim.Optimizer

    def step(
        self,
        loss_closure: LossClosure,
        sharpness_scale: float,
        *,
        step_index: int,
        learning_rate_policy: LearningRatePolicy,
    ) -> UpdateResult: ...


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


def _set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


class GradientUpdate:
    """A standard optimizer update; data batching determines GD versus SGD."""

    def __init__(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        self.optimizer = _build_optimizer(model.parameters(), config.get("optimizer", {}))

    def step(
        self,
        loss_closure: LossClosure,
        sharpness_scale: float = 0.0,
        *,
        step_index: int,
        learning_rate_policy: LearningRatePolicy,
    ) -> UpdateResult:
        if learning_rate_policy.requires_regularized_loss:
            raise ValueError(
                "An objective-dependent learning-rate policy requires algorithm "
                "'s_sam', which estimates the regularized objective during its update"
            )
        learning_rate = learning_rate_policy(step_index, sharpness_scale, None)
        _set_learning_rate(self.optimizer, learning_rate)
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_closure()
        if loss.numel() != 1:
            raise ValueError("The loss closure must return a scalar loss")
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"The clean loss is non-finite: {loss.item()}")
        loss.backward()
        self.optimizer.step()
        value = float(loss.detach().item())
        return UpdateResult(value, learning_rate, value)


class StochasticSharpnessUpdate:
    """S-SAM using online means of perturbed losses and gradients.

    The regularized objective and its gradient are estimated in the same sampling
    loop. Memory therefore stays independent of ``samples``, and an adaptive
    learning-rate policy adds no extra perturbed forward passes.
    """

    def __init__(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        self.model = model
        self.optimizer = _build_optimizer(model.parameters(), config.get("optimizer", {}))
        perturbation = config.get("perturbation", {})
        self.samples = int(perturbation.get("samples", 1))
        self.normalized = bool(perturbation.get("normalized", False))
        self.antithetic = bool(perturbation.get("antithetic", False))
        self.preserve_buffers = bool(perturbation.get("preserve_buffers", True))
        max_grad_norm = perturbation.get("max_grad_norm")
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        distribution = str(perturbation.get("distribution", "gaussian")).lower()
        if distribution != "gaussian":
            raise ValueError("S-SAM currently supports Gaussian perturbations")
        if self.samples < 1:
            raise ValueError("S-SAM requires at least one perturbation sample")
        if self.antithetic and self.samples % 2:
            raise ValueError("Antithetic sampling requires an even sample count")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")

    def step(
        self,
        loss_closure: LossClosure,
        sharpness_scale: float,
        *,
        step_index: int,
        learning_rate_policy: LearningRatePolicy,
    ) -> UpdateResult:
        parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        clean_parameters = [parameter.detach().clone() for parameter in parameters]
        buffers = list(self.model.buffers()) if self.preserve_buffers else []
        clean_buffers = [buffer.detach().clone() for buffer in buffers]

        with torch.no_grad():
            clean_loss_tensor = loss_closure()
        if clean_loss_tensor.numel() != 1:
            raise ValueError("The S-SAM loss closure must return a scalar loss")
        if not torch.isfinite(clean_loss_tensor).item():
            raise FloatingPointError(
                "The clean loss is already non-finite before applying an S-SAM "
                f"perturbation: clean_loss={clean_loss_tensor.item()}"
            )
        clean_loss = float(clean_loss_tensor.item())

        mean_gradients = [torch.zeros_like(parameter) for parameter in parameters]
        mean_loss = 0.0
        perturbations: list[torch.Tensor] | None = None

        try:
            for sample_index in range(self.samples):
                if not self.antithetic or sample_index % 2 == 0:
                    perturbations = sample_parameter_noise(
                        parameters,
                        sharpness_scale,
                        normalized=self.normalized,
                    )
                    sign = 1.0
                else:
                    sign = -1.0

                assert perturbations is not None
                with torch.no_grad():
                    for parameter, clean, perturbation in zip(
                        parameters, clean_parameters, perturbations
                    ):
                        parameter.copy_(clean + sign * perturbation)
                    for buffer, clean in zip(buffers, clean_buffers):
                        buffer.copy_(clean)

                self.optimizer.zero_grad(set_to_none=True)
                loss = loss_closure()
                if loss.numel() != 1:
                    raise ValueError("The S-SAM loss closure must return a scalar loss")
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        "Non-finite perturbed loss encountered at sample "
                        f"{sample_index + 1}/{self.samples}, "
                        f"sharpness_scale={sharpness_scale}"
                    )
                loss.backward()

                mean_weight = 1.0 / (sample_index + 1)
                for mean_gradient, parameter in zip(mean_gradients, parameters):
                    gradient = parameter.grad
                    if gradient is None:
                        mean_gradient.mul_(1.0 - mean_weight)
                        continue
                    if not torch.isfinite(gradient).all().item():
                        raise FloatingPointError(
                            "Non-finite perturbed gradient encountered at sample "
                            f"{sample_index + 1}/{self.samples}, "
                            f"sharpness_scale={sharpness_scale}"
                        )
                    mean_gradient.lerp_(gradient.detach(), mean_weight)

                loss_value = float(loss.detach().item())
                mean_loss += (loss_value - mean_loss) * mean_weight
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        finally:
            with torch.no_grad():
                for parameter, clean in zip(parameters, clean_parameters):
                    parameter.copy_(clean)
                for buffer, clean in zip(buffers, clean_buffers):
                    buffer.copy_(clean)

        learning_rate = learning_rate_policy(step_index, sharpness_scale, mean_loss)
        _set_learning_rate(self.optimizer, learning_rate)
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, mean_gradient in zip(parameters, mean_gradients):
            parameter.grad = mean_gradient
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.optimizer.step()

        return UpdateResult(
            loss=mean_loss,
            learning_rate=learning_rate,
            clean_loss=clean_loss,
            regularized_loss=mean_loss,
        )


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
        raise ValueError(
            f"Unknown algorithm {name!r}. Available: {sorted(_UPDATE_RULES)}"
        ) from exc

