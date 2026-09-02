"""The original mixed-linear experiment, now expressed as a general config case."""

from pathlib import Path

import torch

from ssam import (
    build_dataset,
    build_model,
    plot_training_history,
    train,
)

seed = 12

CONFIG_1 = {
    "model": {
        "name": "3L_2d_dense_linear_ssam",
        "type": "mixed_linear",
        "input_dim": 2,
        "layers": [
            {"type": "dense", "in_dim": 2, "out_dim": 4},
            {"type": "dense", "in_dim": 4, "out_dim": 3},
            {"type": "dense", "in_dim": 3, "out_dim": 2}
        ],
        "activation": "identity",
        "output_activation": "identity",
        "output_reduction": "sum",
        "bias": False,
        "parameter_init": {"type": "uniform", "low": -0.5, "high": 0.5},
    },
    "data": {
        "name": "linear_regression",
        "n_samples": 80,
        "input_dim": 2,
        "target_weights": [3.14159, -1.0],
        "noise_std": 0.05,
        "seed": seed ,
    },
    "training": {
        "algorithm": "s_sam",  # change to "gd" or "sgd"
        "steps": 1000,
        "batch_size": 80,
        "learning_rate": {"name": "constant", "value": 0.01},
        # "learning_rate": {
        #     "name": "strong_descent_diag",
        #     "delta": 0.5,
        #     "safety": 0.95,
        #     "max_lr": 0.1,
        #     "loss_floor": 1e-12,
        # },
        # "learning_rate": {"name": "tamed",
        #                   "type": "sgd",
        #                   "inserted_lr":{
        #     "name": "strong_descent_diag",
        #     "delta": 0.5,
        #     "safety": 0.95,
        #     "max_lr": 0.1,
        #     "loss_floor": 1e-12,
        # }
        #                   },
        #"sharpness_scale": {"name": "constant", "value": 1},
        "sharpness_scale": {"name": "inverse_time","initial": 1,"power": 0.5,"floor": 0.05,},
        "perturbation": {"distribution": "gaussian", "samples": 100},
        "optimizer": {"name": "sgd", "momentum": 0.0},
        "loss": "mse",
        "checkpoint_every": 1000,
        "seed": seed,
        "device": "auto",
    },
}

CONFIG_2 = {
    "model": {
        "name": "3L_2d_dense_linear_gd",
        "type": "mixed_linear",
        "input_dim": 2,
        "layers": [
            {"type": "dense", "in_dim": 2, "out_dim": 4},
            {"type": "dense", "in_dim": 4, "out_dim": 3},
            {"type": "dense", "in_dim": 3, "out_dim": 2}
        ],
        "activation": "identity",
        "output_activation": "identity",
        "output_reduction": "sum",
        "bias": False,
        "parameter_init": {"type": "uniform", "low": -0.5, "high": 0.5},
    },
    "data": {
        "name": "linear_regression",
        "n_samples": 40,
        "input_dim": 2,
        "target_weights": [3.14159, -1.0],
        "noise_std": 0.05,
        "seed": seed ,
    },
    "training": {
        "algorithm": "gd",  # change to "gd" or "sgd"
        "steps": 1000,
        "batch_size": 40,
        "learning_rate": {"name": "constant", "value": 0.01},
        # "learning_rate": {
        #     "name": "strong_descent_diag",
        #     "delta": 0.5,
        #     "safety": 0.95,
        #     "max_lr": 0.1,
        #     "loss_floor": 1e-12,
        # },
        # "learning_rate": {"name": "tamed",
        #                   "type": "sgd",
        #                   "inserted_lr":{
        #     "name": "strong_descent_diag",
        #     "delta": 0.5,
        #     "safety": 0.95,
        #     "max_lr": 0.1,
        #     "loss_floor": 1e-12,
        # }
        #                   },
        #"sharpness_scale": {"name": "constant", "value": 1},
        #"sharpness_scale": {"name": "inverse_time","initial": 1,"power": 0.5,"floor": 0.05,},
        "perturbation": {"distribution": "gaussian", "samples": 100},
        "optimizer": {"name": "sgd", "momentum": 0.0},
        "loss": "mse",
        "checkpoint_every": 1000,
        "seed": seed,
        "device": "auto",
    },
}

def main() -> None:
    output_dir = Path("outputs")
    dataset = build_dataset(CONFIG_1["data"])

    model_1 = build_model(CONFIG_1)
    result_1 = train(model_1, dataset, CONFIG_1)

    model_2 = build_model(CONFIG_2)
    result_2 = train(model_2, dataset, CONFIG_2)

    plot_training_history(result_1, output_dir / f"{CONFIG_1["model"]["name"]}_{CONFIG_1["training"]["algorithm"]}.png")
    plot_training_history(result_2, output_dir / f"{CONFIG_2["model"]["name"]}_{CONFIG_2["training"]["algorithm"]}.png")

    print(f"Final loss with S-SAM: {result_1.history['loss'][-1]:.6f}")
    print(f"Final loss with GD: {result_2.history['loss'][-1]:.6f}")
    print(f"True parameter: {CONFIG_1['data']['target_weights']}")
    print(f"Final S-SAM parameters: {result_1.parameter_snapshots[-1]}")
    print(f"Final GD parameters: {result_2.parameter_snapshots[-1]}")
    #print(f"Plots written to {output_dir.resolve()}")
    # print(f"Initial layer balancedness: {result_1.history["layer_balance"][0]}")
    # print(f"Final layer balancedness: {result_1.history["layer_balance"][-1]}")


if __name__ == "__main__":
    main()

