"""One training loop for GD, SGD, S-SAM, and future update rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .config import training_config
from .schedules import build_learning_rate_policy, build_schedule
from .update_rules import build_update_rule


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainingResult:
    """Training outputs kept small enough for plotting and notebook use."""

    model: nn.Module
    history: dict[str, list[Any]]
    parameter_snapshots: list[torch.Tensor] = field(default_factory=list)
    snapshot_losses: list[float] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def _loss_from_name(name: str) -> LossFunction:
    def align_scalar_output(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if predictions.ndim == targets.ndim + 1 and predictions.shape[-1] == 1:
            return predictions.squeeze(-1)
        return predictions

    losses: dict[str, LossFunction] = {
        "mse": lambda predictions, targets: nn.functional.mse_loss(
            align_scalar_output(predictions, targets), targets
        ),
        "cross_entropy": nn.CrossEntropyLoss(),
        "bce_logits": lambda predictions, targets: (
            nn.functional.binary_cross_entropy_with_logits(
                align_scalar_output(predictions, targets), targets
            )
        ),
        "l1": lambda predictions, targets: nn.functional.l1_loss(
            align_scalar_output(predictions, targets), targets
        ),
    }
    try:
        return losses[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown loss {name!r}. Available: {sorted(losses)}") from exc


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _as_dataset(
    data: Dataset | DataLoader | tuple[torch.Tensor, torch.Tensor],
) -> Dataset:
    if isinstance(data, DataLoader):
        return data.dataset
    if isinstance(data, tuple) and len(data) == 2:
        return TensorDataset(*data)
    if isinstance(data, Dataset):
        return data
    raise TypeError("data must be a Dataset, DataLoader, or (inputs, targets) tuple")


def _make_loader(
    data: Dataset | DataLoader | tuple[torch.Tensor, torch.Tensor],
    config: Mapping[str, Any],
) -> DataLoader:
    if isinstance(data, DataLoader) and config["algorithm"] != "gd":
        return data
    dataset = _as_dataset(data)
    full_batch = config["algorithm"] == "gd"
    batch_size = len(dataset) if full_batch else min(int(config["batch_size"]), len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not full_batch,
        num_workers=int(config.get("num_workers", 0)),
    )


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().reshape(-1).cpu() for parameter in model.parameters()]
    )


def _inferred_policy_shape(
    model: nn.Module, config: Mapping[str, Any]
) -> tuple[int, int]:
    """Infer practical defaults when theorem constants are not configured."""

    model_section = config.get("model", {})
    dimension = int(
        model_section.get(
            "input_dim",
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        )
    )
    if hasattr(model, "layers"):
        depth = len(model.layers)
    else:
        depth = sum(
            1
            for module in model.modules()
            if module is not model
            and any(parameter.requires_grad for parameter in module.parameters(recurse=False))
        )
    return max(1, dimension), max(1, depth)


def _layer_balancedness(model: nn.Module) -> list[float]:
    """Return adjacent-layer Gram-matrix discrepancies where shapes permit."""

    layers = getattr(model, "layers", None)
    if layers is None:
        return []

    values: list[float] = []
    for current, following in zip(layers, layers[1:]):
        if not hasattr(current, "weight") or not hasattr(following, "weight"):
            continue
        current_weight = current.weight.detach()
        following_weight = following.weight.detach()
        if current_weight.ndim == 1:
            current_weight = torch.diag(current_weight)
        if following_weight.ndim == 1:
            following_weight = torch.diag(following_weight)
        left = following_weight.T @ following_weight
        right = current_weight @ current_weight.T
        if left.shape != right.shape:
            values.append(float("nan"))
        else:
            values.append(float(torch.linalg.matrix_norm(left - right).item()))
    return values


def train(
    model: nn.Module,
    data: Dataset | DataLoader | tuple[torch.Tensor, torch.Tensor],
    config: Mapping[str, Any],
    loss_fn: LossFunction | None = None,
    callbacks: list[Callable[[int, nn.Module, dict[str, Any]], None]] | None = None,
) -> TrainingResult:
    """Train using the algorithm and schedules selected in the dictionary.

    ``gd`` always uses the complete dataset. ``sgd`` and ``s_sam`` use the given
    mini-batch size, which naturally becomes full-batch when it equals the dataset
    size. S-SAM estimates its regularized objective and gradient together.
    """

    resolved = training_config(config)
    torch.manual_seed(int(resolved.get("seed", 0)))
    device = _device(str(resolved.get("device", "auto")))
    model.to(device)
    loader = _make_loader(data, resolved)
    loss_function = loss_fn or _loss_from_name(str(resolved.get("loss", "mse")))

    dimension, depth = _inferred_policy_shape(model, config)
    learning_rate_policy = build_learning_rate_policy(
        resolved["learning_rate"],
        default_dimension=dimension,
        default_depth=depth,
    )
    if learning_rate_policy.requires_regularized_loss and resolved["algorithm"] != "s_sam":
        raise ValueError(
            f"Learning-rate policy {learning_rate_policy.name!r} requires algorithm "
            "'s_sam' so its objective estimate and gradient can share samples"
        )
    sharpness = build_schedule(resolved.get("sharpness_scale", 0.0))
    update_rule = build_update_rule(model, resolved)
    callbacks = callbacks or []
    checkpoint_every = int(resolved.get("checkpoint_every", 0))

    history: dict[str, list[Any]] = {
        "step": [],
        "epoch": [],
        "loss": [],
        "clean_loss": [],
        "regularized_loss": [],
        "learning_rate": [],
        "sharpness_scale": [],
        "layer_balance": [],
    }
    result = TrainingResult(model=model, history=history, config=resolved)
    if checkpoint_every:
        result.parameter_snapshots.append(flatten_parameters(model))
        result.snapshot_losses.append(float("nan"))

    target_steps = int(resolved.get("steps", 0))
    epochs = (
        math.ceil(target_steps / max(1, len(loader)))
        if target_steps
        else int(resolved.get("epochs", 1))
    )
    model_section = config.get("model", {})
    model_name = model_section.get("name", model.__class__.__name__)
    model_type = model_section.get("type", model.__class__.__name__)
    print(f"Training {model_name} of type {model_type!r}")
    print(f"Learning rate policy: {learning_rate_policy.name}")

    step = 0
    stop = False
    for epoch in range(epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            scale = float(sharpness(step))

            def closure() -> torch.Tensor:
                return loss_function(model(inputs), targets)

            outcome = update_rule.step(
                closure,
                scale,
                step_index=step,
                learning_rate_policy=learning_rate_policy,
            )
            record: dict[str, Any] = {
                "step": step,
                "epoch": epoch,
                "loss": outcome.loss,
                "clean_loss": outcome.clean_loss,
                "regularized_loss": (
                    float("nan")
                    if outcome.regularized_loss is None
                    else outcome.regularized_loss
                ),
                "learning_rate": outcome.learning_rate,
                "sharpness_scale": scale,
                "layer_balance": _layer_balancedness(model),
            }
            for key, value in record.items():
                history[key].append(value)
            if checkpoint_every and (step + 1) % checkpoint_every == 0:
                result.parameter_snapshots.append(flatten_parameters(model))
                result.snapshot_losses.append(outcome.loss)
            for callback in callbacks:
                callback(step, model, record)

            step += 1
            if target_steps and step >= target_steps:
                stop = True
                break
        if stop:
            break

    if checkpoint_every and step and step % checkpoint_every:
        result.parameter_snapshots.append(flatten_parameters(model))
        result.snapshot_losses.append(float(history["loss"][-1]))
    return result

