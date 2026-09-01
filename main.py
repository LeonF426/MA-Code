"""The original mixed-linear experiment, now expressed as a general config case."""

from pathlib import Path

import torch

from ssam import (
    build_dataset,
    build_model,
    plot_checkpoint_embedding,
    plot_loss_landscape,
    plot_training_history,
    train,
)

seed = 60

CONFIG = {
    "model": {
        "name": "3L_2d_mlp_sig_nob",
        "type": "mlp",
        "input_dim": 2,
        "layers": [
            {"type": "dense", "in_dim": 2, "out_dim": 3},
            {"type": "dense", "in_dim": 3, "out_dim": 4},
            {"type": "dense", "in_dim": 4, "out_dim": 2}
        ],
        "activation": "sigmoid",
        "output_activation": "sigmoid",
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
        "algorithm": "s_sam",  # change to "gd" or "sgd"
        "steps": 500,
        "batch_size": 40,
        "learning_rate": {"name": "constant", "value": 0.01},
        #"learning_rate": {
        #    "name": "strong_descent_diag",
        #    "delta": 0.5,
        #    "safety": 0.95,
        #    "max_lr": 0.1,
        #    "loss_floor": 1e-12,
        #},
        "sharpness_scale": {
            "name": "inverse_time",
            "initial": 1,
            "power": 0.25,
            "floor": 0.0,
        },
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
    dataset = build_dataset(CONFIG["data"])
    model = build_model(CONFIG)
    result = train(model, dataset, CONFIG)

    plot_training_history(result, output_dir / f"{CONFIG["model"]["name"]}_{CONFIG["training"]["algorithm"]}.png")

    print(f"Final loss: {result.history['loss'][-1]:.6f}")
    print(f"True parameter: {CONFIG['data']['target_weights']}")
    print(f"Final parameters: {result.parameter_snapshots[-1]}")
    print(f"Plots written to {output_dir.resolve()}")
    print(f"Initial layer balancedness: {result.history["layer_balance"][0]}")
    print(f"Final layer balancedness: {result.history["layer_balance"][-1]}")


if __name__ == "__main__":
    main()

