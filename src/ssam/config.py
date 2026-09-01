"""Dictionary-first experiment configuration and validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": "model",
        "type": "mlp",
        "input_dim": 1,
        "output_dim": 1,
        "depth": 2,
        "width": 64,
        "activation": "relu",
        "output_activation": "identity",
        "bias": True,
        "parameter_init": {"type": "xavier_uniform"},
    },
    "training": {
        "algorithm": "sgd",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": {"name": "constant", "value": 1e-2},
        "sharpness_scale": {"name": "constant", "value": 0.0},
        "optimizer": {"name": "sgd", "momentum": 0.0, "weight_decay": 0.0},
        "loss": "mse",
        "seed": 0,
        "device": "auto",
        "checkpoint_every": 0,
    },
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated config with defaults filled in.

    The input and output deliberately remain plain dictionaries so experiments are
    easy to serialize, copy, and alter from notebooks or configuration files.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    unknown = set(config) - {"model", "training", "data", "visualization"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")

    normalized = _deep_merge(DEFAULT_CONFIG, config)
    model = normalized["model"]
    training = normalized["training"]

    model_override = config.get("model", {})
    model_type = str(model.get("type", "")).lower()
    if isinstance(model_override, Mapping) and "type" not in model_override:
        legacy_name = str(model_override.get("name", "")).lower()
        if legacy_name in {"mlp", "mixed_linear"} or legacy_name.startswith(
            "torchvision/"
        ):
            model_type = legacy_name
        elif "name" in model_override:
            raise ValueError(
                "A free-form model.name requires model.type (for example, 'mlp')"
            )
    if isinstance(model_override, Mapping):
        init_override = model_override.get("parameter_init")
        if (
            isinstance(init_override, Mapping)
            and "type" not in init_override
            and "name" in init_override
        ):
            model["parameter_init"]["type"] = init_override["name"]
    if not model_type:
        legacy_name = str(model.get("name", "")).lower()
        if legacy_name in {"mlp", "mixed_linear"} or legacy_name.startswith(
            "torchvision/"
        ):
            model_type = legacy_name
        else:
            raise ValueError(
                "model.type is required; model.name is a free-form experiment label"
            )
    model["type"] = model_type

    if model_type in {"mlp", "mixed_linear"}:
        if int(model.get("input_dim", 0)) < 1:
            raise ValueError("model.input_dim must be positive")
        if "layers" not in model and int(model.get("depth", 0)) < 1:
            raise ValueError("model.depth must be at least 1")
    elif not model_type.startswith("torchvision/"):
        raise ValueError(
            "model.type must be 'mlp', 'mixed_linear', or 'torchvision/<model>'"
        )

    algorithm = str(training.get("algorithm", "")).lower().replace("-", "_")
    if algorithm == "ssam":
        algorithm = "s_sam"
    training["algorithm"] = algorithm
    if int(training.get("epochs", 0)) < 1 and int(training.get("steps", 0)) < 1:
        raise ValueError("training.epochs or training.steps must be positive")
    if algorithm != "gd" and int(training.get("batch_size", 0)) < 1:
        raise ValueError("training.batch_size must be positive")
    if algorithm == "s_sam":
        perturbation = training.setdefault("perturbation", {})
        perturbation.setdefault("distribution", "gaussian")
        perturbation.setdefault("samples", 1)
        perturbation.setdefault("normalized", False)
        perturbation.setdefault("antithetic", False)
        perturbation.setdefault("preserve_buffers", True)
        if int(perturbation["samples"]) < 1:
            raise ValueError("training.perturbation.samples must be positive")
        if bool(perturbation["antithetic"]) and int(perturbation["samples"]) % 2:
            raise ValueError(
                "training.perturbation.samples must be even with antithetic sampling"
            )

    return normalized


def model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either a full experiment config or its ``model`` section."""

    if "model" in config:
        return normalize_config(config)["model"]
    return normalize_config({"model": config})["model"]


def training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either a full experiment config or its ``training`` section."""

    if "training" in config:
        return normalize_config(config)["training"]
    return normalize_config({"training": config})["training"]

