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
from .schedules import build_schedule, strong_descent_diag
from .update_rules import build_update_rule


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainingResult:
    """Training outputs kept small enough for plotting and notebook use."""

    model: nn.Module
    history: dict[str, list[float | int]]
    parameter_snapshots: list[torch.Tensor] = field(default_factory=list)
    snapshot_losses: list[float] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def _loss_from_name(name: str) -> LossFunction:
    def align_scalar_output(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.ndim == targets.ndim + 1 and predictions.shape[-1] == 1:
            return predictions.squeeze(-1)
        return predictions

    losses: dict[str, LossFunction] = {
        "mse": lambda predictions, targets: nn.functional.mse_loss(
            align_scalar_output(predictions, targets), targets
        ),
        "cross_entropy": nn.CrossEntropyLoss(),
        "bce_logits": lambda predictions, targets: nn.functional.binary_cross_entropy_with_logits(
            align_scalar_output(predictions, targets), targets
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


def _as_dataset(data: Dataset | DataLoader | tuple[torch.Tensor, torch.Tensor]) -> Dataset:
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
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])


def train(
    model: nn.Module,
    data: Dataset | DataLoader | tuple[torch.Tensor, torch.Tensor],
    config: Mapping[str, Any],
    loss_fn: LossFunction | None = None,
    callbacks: list[Callable[[int, nn.Module, dict[str, Any]], None]] | None = None,
) -> TrainingResult:
    """Train a model using the algorithm selected in the input dictionary.

    ``gd`` uses one full-dataset batch, ``sgd`` uses mini-batches, and ``s_sam``
    uses the same mini-batches with gradients evaluated at random perturbations.
    """

    resolved = training_config(config)
    torch.manual_seed(int(resolved.get("seed", 0)))
    device = _device(str(resolved.get("device", "auto")))
    model.to(device)
    loader = _make_loader(data, resolved)
    loss_function = loss_fn or _loss_from_name(str(resolved.get("loss", "mse")))

    if config["training"]["learning_rate"]["name"] == "strong_descent_diag":
        learning_rate = strong_descent_diag(
            dimension=config["model"]["input_dim"],
            depth=len(config["model"]["layers"]),
            delta=config["training"]["learning_rate"]["delta"],
            safety= config["training"]["learning_rate"]["safety"],
            max_lr=config["training"]["learning_rate"]["max_lr"]
        )
    else:
        learning_rate = build_schedule(resolved["learning_rate"])
    print("building schedule worked")

    sharpness = build_schedule(resolved.get("sharpness_scale", 0.0))
    update_rule = build_update_rule(model, resolved)
    callbacks = callbacks or []
    checkpoint_every = int(resolved.get("checkpoint_every", 0))

    history: dict[str, list[float | int]] = {
        "step": [],
        "epoch": [],
        "loss": [],
        "learning_rate": [],
        "sharpness_scale": [],
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
    print(f"Training {config["model"]["name"]} of type '{config["model"]["type"]}'")
    print(f"Learning rate schedule: {config["training"]["learning_rate"]["name"]}")
    step = 0
    stop = False
    for epoch in range(epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            scale = float(sharpness(step))

            if config["training"]["learning_rate"]["name"] == "strong_descent_diag":
                with torch.no_grad():
                    reg_loss_value = regularized_objective()

                    lr = learning_rate(step,scale,reg_loss_value)
            else:
                lr = float(learning_rate(step))


            for group in update_rule.optimizer.param_groups:
                group["lr"] = lr

            def closure() -> torch.Tensor:
                return loss_function(model(inputs), targets)

            loss = update_rule.step(closure, scale)
            record = {
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "learning_rate": lr,
                "sharpness_scale": scale,
            }
            for key, value in record.items():
                history[key].append(value)
            if checkpoint_every and (step + 1) % checkpoint_every == 0:
                result.parameter_snapshots.append(flatten_parameters(model))
                result.snapshot_losses.append(loss)
            for callback in callbacks:
                callback(step, model, record)

            step += 1
            if target_steps and step >= target_steps:
                stop = True
                break
        if stop:
            break

    return result
