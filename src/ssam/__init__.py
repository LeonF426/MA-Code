"""A compact toolkit for configurable GD, SGD, and stochastic S-SAM experiments."""

from .config import DEFAULT_CONFIG, normalize_config
from .data import build_dataset, make_linear_regression_dataset
from .layers import DiagLinear
from .models import ConfigurableNet, MixedLinearNet, build_model, register_activation
from .schedules import build_schedule, register_schedule
from .trainers import TrainingResult, train
from .update_rules import register_update_rule
from .visualization import (
    plot_checkpoint_embedding,
    plot_loss_landscape,
    plot_training_history,
)

__all__ = [
    "DEFAULT_CONFIG",
    "ConfigurableNet",
    "DiagLinear",
    "MixedLinearNet",
    "TrainingResult",
    "build_dataset",
    "build_model",
    "build_schedule",
    "make_linear_regression_dataset",
    "normalize_config",
    "plot_checkpoint_embedding",
    "plot_loss_landscape",
    "plot_training_history",
    "register_activation",
    "register_schedule",
    "register_update_rule",
    "train",
]
