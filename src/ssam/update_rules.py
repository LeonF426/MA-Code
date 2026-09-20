"""Extensible parameter-update rules, including stochastic sharpness-aware SGD."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import math
import torch
from torch import nn

from .objectives import sample_parameter_noise
from .schedules import LearningRatePolicy


LossValue = torch.Tensor | Mapping[str, torch.Tensor]
LossClosure = Callable[[], LossValue]


def _split_loss(value: LossValue) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the scalar objective and any scalar diagnostic components.

    Scalar closures remain the default. Structured closures let derivative-based
    objectives such as PINNs report their constituent losses without repeating
    their relatively expensive coordinate-derivative calculation.
    """

    if isinstance(value, torch.Tensor):
        return value, {}
    if not isinstance(value, Mapping):
        raise TypeError("The loss closure must return a tensor or a mapping")
    objective_key = "loss" if "loss" in value else "total_loss"
    if objective_key not in value:
        raise ValueError("A structured loss must contain 'loss' or 'total_loss'")
    loss = value[objective_key]
    components = {key: component for key, component in value.items() if key != objective_key}
    if not isinstance(loss, torch.Tensor) or any(
        not isinstance(component, torch.Tensor) for component in components.values()
    ):
        raise TypeError("All structured loss values must be tensors")
    return loss, components


def _scalar_values(components: Mapping[str, torch.Tensor]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, component in components.items():
        if component.numel() != 1:
            raise ValueError(f"Loss component {name!r} must be scalar")
        if not torch.isfinite(component).item():
            raise FloatingPointError(f"Loss component {name!r} is non-finite")
        values[name] = float(component.detach().item())
    return values


@dataclass(frozen=True)
class UpdateResult:
    """Values produced by one parameter update."""

    loss: float
    learning_rate: float
    clean_loss: float
    regularized_loss: float | None = None
    clean_components: dict[str, float] = field(default_factory=dict)
    regularized_components: dict[str, float] = field(default_factory=dict)


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


def _global_gradient_norm(
    parameters: list[nn.Parameter],
) -> float:
    """Return the global L2 norm of the installed gradients."""

    squared_norm = 0.0

    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += float(
                parameter.grad.detach().square().sum().item()
            )

    gradient_norm = math.sqrt(squared_norm)

    if not math.isfinite(gradient_norm):
        raise FloatingPointError(
            f"Non-finite gradient norm: {gradient_norm}"
        )

    return gradient_norm


class GradientUpdate:
    """A standard optimizer update; data batching determines GD versus SGD."""

    def __init__(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        self.model = model
        self.optimizer = _build_optimizer(model.parameters(), config.get("optimizer", {}))
        max_grad_norm = config.get("max_grad_norm")
        self.max_grad_norm = (
            None
            if max_grad_norm is None
            else float(max_grad_norm)
        )

    def step(
        self,
        loss_closure: LossClosure,
        sharpness_scale: float = 0.0,
        *,
        step_index: int,
        learning_rate_policy: LearningRatePolicy,
    ) -> UpdateResult:

        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        # Compute the ordinary GD/SGD gradient.
        self.optimizer.zero_grad(set_to_none=True)

        loss, _ = _split_loss(loss_closure())

        if loss.numel() != 1:
            raise ValueError(
                "The loss closure must return a scalar loss"
            )

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite loss at step {step_index}: {loss.item()}"
            )

        loss.backward()

        # If clipping is enabled, apply it before measuring the norm so that
        # the tamed policy uses the gradient actually sent to the optimizer.
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                self.max_grad_norm,
            )

        # For SGD, this is the norm of the gradient estimated from the current
        # training mini-batch. No explicit batch-size term is required.

        gradient_norm = None
        if learning_rate_policy.requires_gradient_norm:
            gradient_norm = _global_gradient_norm(parameters)


        if learning_rate_policy.requires_regularized_loss:
            raise ValueError(
                "An objective-dependent learning-rate policy requires algorithm "
                "'s_sam', which estimates the regularized objective during its update"
            )
        learning_rate = learning_rate_policy(step_index, sharpness_scale,gradient_norm=gradient_norm )
        _set_learning_rate(self.optimizer, learning_rate)
        self.optimizer.zero_grad(set_to_none=True)
        loss, clean_component_tensors = _split_loss(loss_closure())
        if loss.numel() != 1:
            raise ValueError("The loss closure must return a scalar loss")
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"The clean loss is non-finite: {loss.item()}")
        loss.backward()
        self.optimizer.step()
        value = float(loss.detach().item())
        return UpdateResult(
            value,
            learning_rate,
            value,
            clean_components=_scalar_values(clean_component_tensors),
        )


class StochasticSharpnessUpdate:
    """Stochastic sharpness-aware minimization using online averages.

    For every optimization step, Gaussian perturbations are applied to the
    current parameters. The perturbed losses and gradients are averaged without
    retaining every individual sample.

    ``samples`` is the number of Gaussian parameter perturbations. It is not
    the dataset mini-batch size; the mini-batch is supplied through
    ``loss_closure`` by the DataLoader.

    A tamed learning-rate policy applies

        effective_lr = alpha / (1 + alpha * ||g||),

    where ``alpha`` is produced by the inserted policy and ``g`` is the final
    averaged gradient used by the optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
    ) -> None:
        self.model = model

        # For the update to exactly match tamed SGD, this should construct plain
        # SGD without momentum or weight decay.
        self.optimizer = _build_optimizer(
            model.parameters(),
            config.get("optimizer", {}),
        )

        perturbation = config.get("perturbation", {})

        # Number of Monte Carlo samples used to approximate the Gaussian
        # expectation defining the regularized objective.
        self.samples = int(perturbation.get("samples", 1))

        # Controls how sample_parameter_noise constructs each perturbation.
        self.normalized = bool(perturbation.get("normalized", False))

        # When enabled, perturbations are used in pairs: z and -z.
        self.antithetic = bool(perturbation.get("antithetic", False))

        # Restore model buffers such as BatchNorm running statistics after
        # evaluating perturbed models.
        self.preserve_buffers = bool(
            perturbation.get("preserve_buffers", True)
        )

        # PINN losses need autograd enabled even when only their clean value is
        # logged, because spatial derivatives are part of the objective. The
        # default stays False for existing supervised applications.
        self.loss_requires_grad = bool(config.get("loss_requires_grad", False))

        # Optional clipping is applied before calculating the norm used by the
        # tamed policy. Thus, when clipping is enabled, g denotes the clipped
        # gradient actually passed to the optimizer.
        max_grad_norm = perturbation.get("max_grad_norm")
        self.max_grad_norm = (
            None if max_grad_norm is None else float(max_grad_norm)
        )

        distribution = str(
            perturbation.get("distribution", "gaussian")
        ).lower()

        if distribution != "gaussian":
            raise ValueError(
                "S-SAM currently supports Gaussian perturbations"
            )

        if self.samples < 1:
            raise ValueError(
                "S-SAM requires at least one perturbation sample"
            )

        if self.antithetic and self.samples % 2:
            raise ValueError(
                "Antithetic sampling requires an even sample count"
            )

        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")

    @staticmethod
    def _global_gradient_norm(
        parameters: list[nn.Parameter],
    ) -> float:
        """Calculate the global Euclidean norm of all parameter gradients."""

        squared_norm = 0.0

        for parameter in parameters:
            if parameter.grad is not None:
                squared_norm += float(
                    parameter.grad.detach().square().sum().item()
                )

        gradient_norm = math.sqrt(squared_norm)

        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite averaged gradient norm: {gradient_norm}"
            )

        return gradient_norm

    def step(
        self,
        loss_closure: LossClosure,
        sharpness_scale: float,
        *,
        step_index: int,
        learning_rate_policy: LearningRatePolicy,
    ) -> UpdateResult:
        # Only trainable parameters participate in perturbation and updating.
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        if not parameters:
            raise ValueError("The model has no trainable parameters")

        # Save the unperturbed parameter values. Every Gaussian sample must be
        # evaluated relative to this same central parameter vector.
        clean_parameters = [
            parameter.detach().clone()
            for parameter in parameters
        ]

        # Forward passes can modify buffers, especially BatchNorm statistics.
        # Save them so perturbation samples do not alter the model state.
        buffers = (
            list(self.model.buffers())
            if self.preserve_buffers
            else []
        )
        clean_buffers = [
            buffer.detach().clone()
            for buffer in buffers
        ]

        # Evaluate the ordinary, unperturbed loss for logging. Gradients are not
        # needed here because the update uses the regularized gradient below.
        try:
            with torch.set_grad_enabled(self.loss_requires_grad):
                clean_loss_tensor, clean_component_tensors = _split_loss(loss_closure())
        finally:
            # Undo any buffer changes caused by the clean forward pass.
            with torch.no_grad():
                for buffer, clean_buffer in zip(
                    buffers,
                    clean_buffers,
                ):
                    buffer.copy_(clean_buffer)

        if clean_loss_tensor.numel() != 1:
            raise ValueError(
                "The S-SAM loss closure must return a scalar loss"
            )

        if not torch.isfinite(clean_loss_tensor).item():
            raise FloatingPointError(
                "The clean loss is already non-finite before applying an "
                "S-SAM perturbation: "
                f"clean_loss={clean_loss_tensor.item()}"
            )

        clean_loss = float(clean_loss_tensor.item())
        clean_components = _scalar_values(clean_component_tensors)
        del clean_loss_tensor, clean_component_tensors

        # Online means avoid storing one complete gradient for every Gaussian
        # sample. Memory use therefore does not grow with self.samples.
        mean_gradients = [
            torch.zeros_like(parameter)
            for parameter in parameters
        ]
        mean_loss = 0.0
        mean_components: dict[str, float] = {}

        # In antithetic mode, this holds z while both +z and -z are evaluated.
        perturbations: list[torch.Tensor] | None = None

        try:
            for sample_index in range(self.samples):
                # Draw a new perturbation for ordinary sampling and for the
                # first member of each antithetic pair.
                if not self.antithetic or sample_index % 2 == 0:
                    perturbations = sample_parameter_noise(
                        parameters,
                        sharpness_scale,
                        normalized=self.normalized,
                    )
                    sign = 1.0
                else:
                    # Reuse the preceding perturbation with the opposite sign.
                    sign = -1.0

                assert perturbations is not None

                # Evaluate the sample at theta + eta*z or theta - eta*z.
                with torch.no_grad():
                    for parameter, clean, perturbation in zip(
                        parameters,
                        clean_parameters,
                        perturbations,
                    ):
                        parameter.copy_(
                            clean + sign * perturbation
                        )

                    # Each sample starts from identical non-parameter state.
                    for buffer, clean_buffer in zip(
                        buffers,
                        clean_buffers,
                    ):
                        buffer.copy_(clean_buffer)

                self.optimizer.zero_grad(set_to_none=True)

                # This is a perturbed evaluation of the loss. Calling backward
                # produces a sample of the regularized-objective gradient.
                loss, component_tensors = _split_loss(loss_closure())
                # print(f"evaluated loss in SSAM update: {loss.item()}")

                if loss.numel() != 1:
                    raise ValueError(
                        "The S-SAM loss closure must return a scalar loss"
                    )

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        "Non-finite perturbed loss encountered at sample "
                        f"{sample_index + 1}/{self.samples}, "
                        f"sharpness_scale={sharpness_scale}"
                    )

                loss.backward()

                # Given n samples and the previous mean m, lerp with weight 1/n
                # performs the online update
                #
                #     new_mean = old_mean + (sample - old_mean) / n.
                mean_weight = 1.0 / (sample_index + 1)

                for mean_gradient, parameter in zip(
                    mean_gradients,
                    parameters,
                ):
                    gradient = parameter.grad

                    if gradient is None:
                        # Treat a missing sample gradient as zero.
                        mean_gradient.mul_(1.0 - mean_weight)
                        continue

                    if not torch.isfinite(gradient).all().item():
                        raise FloatingPointError(
                            "Non-finite perturbed gradient encountered at "
                            f"sample {sample_index + 1}/{self.samples}, "
                            f"sharpness_scale={sharpness_scale}"
                        )

                    mean_gradient.lerp_(
                        gradient.detach(),
                        mean_weight,
                    )

                # Update the regularized-loss estimate using the same Gaussian
                # sample that contributed the gradient above.
                loss_value = float(loss.detach().item())
                mean_loss += (
                    loss_value - mean_loss
                ) * mean_weight
                component_values = _scalar_values(component_tensors)
                if sample_index == 0:
                    mean_components = dict(component_values)
                else:
                    if component_values.keys() != mean_components.keys():
                        raise ValueError(
                            "Structured loss components must be identical for every sample"
                        )
                    for name, value in component_values.items():
                        mean_components[name] += (
                            value - mean_components[name]
                        ) * mean_weight
                # print(f"Mean loss : {mean_loss}")

        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise

        finally:
            # Parameter updates must be applied at theta, not at the final
            # perturbed position theta + eta*z.
            with torch.no_grad():
                for parameter, clean in zip(
                    parameters,
                    clean_parameters,
                ):
                    parameter.copy_(clean)

                for buffer, clean_buffer in zip(
                    buffers,
                    clean_buffers,
                ):
                    buffer.copy_(clean_buffer)

        # Install the Monte Carlo mean gradient on the restored parameters.
        self.optimizer.zero_grad(set_to_none=True)

        for parameter, mean_gradient in zip(
            parameters,
            mean_gradients,
        ):
            parameter.grad = mean_gradient

        # If clipping is configured, clip before measuring ||g||. This makes
        # the tamed denominator use the gradient actually applied below.
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                self.max_grad_norm,
            )

        gradient_norm = self._global_gradient_norm(parameters)

        # For a tamed policy this first evaluates inserted_lr, for example
        # strong_descent_diag, to obtain alpha_k and then returns
        #
        #     alpha_k / (1 + alpha_k * gradient_norm).
        #
        # Ordinary policies accept this fourth argument but do not use it.
        learning_rate = learning_rate_policy(
            step_index,
            sharpness_scale,
            mean_loss,
            gradient_norm=gradient_norm,
        )

        _set_learning_rate(
            self.optimizer,
            learning_rate,
        )

        # With plain SGD, this performs:
        #
        # theta <- theta - learning_rate * mean_gradient.
        #
        # Because learning_rate is already tamed, this is the desired update.
        self.optimizer.step()

        return UpdateResult(
            loss=mean_loss,
            learning_rate=learning_rate,
            clean_loss=clean_loss,
            regularized_loss=mean_loss,
            clean_components=clean_components,
            regularized_components=mean_components,
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

