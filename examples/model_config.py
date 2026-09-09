
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ssam import build_dataset, build_model, plot_training_history, train

SEED = 100

BASE_CONFIG = {
    "model": {
        "name": "california_diag",
        "type": "mixed_linear",
        "input_dim": 8,
        "layers": [],  # filled separately for every depth
        "activation": "identity",
        "output_activation": "identity",
        "output_reduction": "sum",
        "bias": False,
        "parameter_init": {
            "type": "identity",
            "bias": 0.0,
        },
    },
    "data": {
        "name": "california_housing",
        "root": "data",
        "test_fraction": 0.2,
        "standardize": True,
        "standardize_target": False,
        "download": True,
        "seed": 7,
    },
    "training": {
        "algorithm": "sgd",  # replaced for every run
        "steps": 500,
        "batch_size": 256,
        "learning_rate": {
            "name": "tamed",
            "type": "sgd",
            "inserted_lr": {
                "name": "constant",
                "value": 0.1,
            },
        },
        "sharpness_scale": {"name": "inverse_time","initial": 2,"power": 0.25,"floor": 0.0},
        "perturbation": {
            "distribution": "gaussian",
            "samples": 8,
            "normalized": True,
            "antithetic": True,
        },
        "optimizer": {
            "name": "sgd",
            "momentum": 0.0,
        },
        "loss": "mse",
        "checkpoint_every": 0,
        "device": "auto",
    },
}

def config_for(depth: int, algorithm: str) -> dict:
    config = copy.deepcopy(BASE_CONFIG)

    config["model"]["name"] = (
        f"california_{depth}L_diag_{algorithm}"
    )

    config["model"]["layers"] = [
        {
            "type": "diag",
            "in_dim": config["model"]["input_dim"],
            "out_dim": config["model"]["input_dim"],
            "bias": False,
        }
        for _ in range(depth)
    ]

    config["training"]["algorithm"] = algorithm

    if algorithm == "sgd":
        config["training"]["sharpness_scale"] = {
            "name": "constant",
            "value": 0.0,
        }

    return config


def model_config(kind: str) -> dict:
    """Return a linear model or a small nonlinear baseline."""

    common = {
        "name": f"california_{kind}",
        "type": "mixed_linear",
        "input_dim": 8,
        "output_dim": 1,
        "output_activation": "identity",
        "bias": True,
        "parameter_init": {"type": "xavier_uniform"},
    }
    if kind == "linear":
        return {**common, "depth": 1, "activation": "identity"}
    return {
        **common,
        "depth": 3,
        "width": [64, 32],
        "activation": "gelu",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("sgd", "s_sam"),
        default=("sgd", "s_sam"),
    )
    parser.add_argument("--model", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--sharpness-scale", type=float, default=0.05)
    parser.add_argument("--perturbation-samples", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/california"))
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require the dataset to exist in scikit-learn's local cache.",
    )
    return parser.parse_args()


