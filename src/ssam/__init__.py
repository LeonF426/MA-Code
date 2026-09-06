"""A compact toolkit for configurable GD, SGD, and stochastic S-SAM experiments."""

from .config import DEFAULT_CONFIG, normalize_config
from .data import (
    CALIFORNIA_HOUSING_FEATURES,
    build_dataset,
    make_california_housing_dataset,
    make_linear_regression_dataset,
    prepare_california_housing_data,
)
from .layers import DiagLinear
from .models import ConfigurableNet, MixedLinearNet, build_model, register_activation
from .objectives import estimate_regularized_objective
from .schedules import (
    LearningRatePolicy,
    build_learning_rate_policy,
    build_schedule,
    register_schedule,
)
from .trainers import TrainingResult, train
from .update_rules import UpdateResult, register_update_rule
from .visualization import (
    plot_training_history,
)
from .average_sharpness import evaluate_average_sharpness

__all__ = [
    "DEFAULT_CONFIG",
    "CALIFORNIA_HOUSING_FEATURES",
    "ConfigurableNet",
    "DiagLinear",
    "MixedLinearNet",
    "LearningRatePolicy",
    "TrainingResult",
    "UpdateResult",
    "build_dataset",
    "build_model",
    "build_learning_rate_policy",
    "build_schedule",
    "estimate_regularized_objective",
    "make_linear_regression_dataset",
    "make_california_housing_dataset",
    "normalize_config",
    "plot_training_history",
    "prepare_california_housing_data",
    "register_activation",
    "register_schedule",
    "register_update_rule",
    "train",
]


