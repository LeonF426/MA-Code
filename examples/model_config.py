"""Configurations for the California Housing comparison.

The targets stay in scikit-learn's original $100,000 unit. Consequently every
model must contain an intercept: centered inputs plus an intercept-free model
cannot reproduce the non-zero target mean.
"""

from __future__ import annotations

import copy


SEED = 102

DATA_CONFIG = {
    "name": "california_housing",
    "root": "data",
    "test_fraction": 0.2,
    # The canonical dataset has no missing values, but keeping this enabled makes
    # preprocessing safe for locally modified copies. Statistics are learned
    # from the training split only.
    "impute_missing": True,
    # A few ratio features contain extreme values. Winsorization prevents those
    # observations from dominating the mean/std and stochastic gradients.
    "clip_quantiles": [0.005, 0.995],
    "standardize": True,
    "standardize_target": False,
    "download": True,
    "seed": 7,
}

TRAINING_CONFIG = {
    "algorithm": "sgd",  # replaced for every matched pair
    "steps": 1000,
    "batch_size": 256,
    "learning_rate": {
        "name": "tamed",
        "type": "sgd",
        "inserted_lr": {"name": "constant", "value": 0.05},
    },
    # A radius of 2 was larger than the useful parameter scale for these models.
    # Normalized noise with radius 0.05 is a deliberately modest regularizer.
    "sharpness_scale": {"name": "constant", "value": 0.05},
    "perturbation": {
        "distribution": "gaussian",
        "samples": 8,
        "normalized": True,
        "antithetic": True,
    },
    "optimizer": {"name": "sgd", "momentum": 0.0},
    "loss": "mse",
    "checkpoint_every": 0,
    "seed": SEED,
    "device": "auto",
}

VISUALIZATION_CONFIG = {
    "output_dir": "outputs/california",
    "interpolation": {
        "sharpness_scale": 0.05,
        "sharpness_samples": 64,
        "sharpness_seed": 10023,
        "interpolation_points": 11,
        "normalized": True,
        "antithetic": True,
    },
}


def _experiment(model: dict) -> dict:
    return {
        "model": model,
        "data": copy.deepcopy(DATA_CONFIG),
        "training": copy.deepcopy(TRAINING_CONFIG),
        "visualization": copy.deepcopy(VISUALIZATION_CONFIG),
    }


# A genuine ordinary linear-regression baseline: one 8 -> 1 affine layer.
BASE_CONFIG_LINEAR = _experiment(
    {
        "name": "california_linear",
        "type": "mixed_linear",
        "input_dim": 8,
        "layers": [{"type": "dense", "out_dim": 1, "bias": True}],
        "activation": "identity",
        "output_activation": "identity",
        "output_reduction": "none",
        "bias": True,
        "parameter_init": {"type": "xavier_uniform", "bias": 0.0},
    }
)


# Filled by config_for for the requested depth. Biases are required because the
# target is intentionally not centered.
BASE_CONFIG = _experiment(
    {
        "name": "california_diag",
        "type": "mixed_linear",
        "input_dim": 8,
        "layers": [],
        "activation": "identity",
        "output_activation": "identity",
        "output_reduction": "sum",
        "bias": True,
        "parameter_init": {
            "type": "identity",
            "bias": 0.0,
            "rescaling": {
                "enabled": True,
                "mode": "layerwise",
                "log_scale_std": 0.5,
                "seed": SEED,
            },
        },
    }
)


# A factorized linear model, kept separate from the ordinary linear baseline.
# The final layer produces one scalar directly; an all-ones initialization and an
# 8-vector output followed by sum made the old starting function unnecessarily
# large and ill-conditioned.
BASE_CONFIG_DENSE = _experiment(
    {
        "name": "california_dense_factorized",
        "type": "mixed_linear",
        "input_dim": 8,
        "layers": [
            {"type": "dense", "out_dim": 16, "bias": False},
            {"type": "dense", "out_dim": 1, "bias": True},
        ],
        "activation": "identity",
        "output_activation": "identity",
        "output_reduction": "none",
        "bias": True,
        "parameter_init": {
            "type": "xavier_uniform",
            "bias": 0.0,
            "rescaling": {
                "enabled": True,
                "mode": "neuronwise",
                "log_scale_std": 0.5,
                "seed": SEED,
            },
        },
    }
)


BASE_CONFIG_RELU = _experiment(
    {
        "name": "california_relu",
        "type": "mixed_linear",
        "input_dim": 8,
        "layers": [
            {"type": "dense", "out_dim": 64, "bias": True},
            {"type": "dense", "out_dim": 32, "bias": True},
            {"type": "dense", "out_dim": 1, "bias": True},
        ],
        "activation": "relu",
        "output_activation": "identity",
        "output_reduction": "none",
        "bias": True,
        "parameter_init": {
            "type": "kaiming_uniform",
            "nonlinearity": "relu",
            "bias": 0.0,
            "rescaling": {
                "enabled": True,
                "mode": "neuronwise",
                "log_scale_std": 0.5,
                "seed": SEED,
            },
        },
    }
)


def config_for(depth: int, algorithm: str) -> dict:
    """Build a matched diagonal-model configuration."""

    if depth < 1:
        raise ValueError("depth must be positive")
    if algorithm not in {"sgd", "s_sam"}:
        raise ValueError("algorithm must be 'sgd' or 's_sam'")

    config = copy.deepcopy(BASE_CONFIG)
    config["model"]["name"] = f"california_{depth}L_diag_{algorithm}"
    config["model"]["layers"] = [
        {"type": "diag", "out_dim": 8, "bias": index == depth - 1}
        for index in range(depth)
    ]
    config["training"]["algorithm"] = algorithm
    if algorithm == "sgd":
        config["training"]["sharpness_scale"] = {
            "name": "constant",
            "value": 0.0,
        }
    return config


def model_config(kind: str) -> dict:
    """Return a copy of the ordinary linear or nonlinear model section."""

    choices = {
        "linear": BASE_CONFIG_LINEAR,
        "mlp": BASE_CONFIG_RELU,
    }
    try:
        return copy.deepcopy(choices[kind]["model"])
    except KeyError as exc:
        raise ValueError("kind must be 'linear' or 'mlp'") from exc
