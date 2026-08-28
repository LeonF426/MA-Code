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


CONFIG = {
    "model": {
        "name": "mixed_linear",
        "input_dim": 2,
        "layers": [
            {"type": "diag", "out_dim": 2},
            {"type": "diag", "out_dim": 2},
        ],
        "activation": "identity",
        "output_activation": "identity",
        "output_reduction": "sum",
        "bias": False,
        "parameter_init": {"name": "uniform", "low": -0.5, "high": 0.5},
    },
    "data": {
        "name": "linear_regression",
        "n_samples": 1024,
        "input_dim": 2,
        "target_weights": [3.14159, -1.0],
        "noise_std": 0.05,
        "seed": 7,
    },
    "training": {
        "algorithm": "s_sam",  # change to "gd" or "sgd"
        "steps": 120,
        "batch_size": 128,
        "learning_rate": {"name": "constant", "value": 0.03},
        "sharpness_scale": {
            "name": "inverse_time",
            "initial": 0.25,
            "power": 0.5,
            "floor": 0.01,
        },
        "perturbation": {"distribution": "gaussian", "samples": 2},
        "optimizer": {"name": "sgd", "momentum": 0.0},
        "loss": "mse",
        "checkpoint_every": 5,
        "seed": 7,
        "device": "auto",
    },
}


def main() -> None:
    output_dir = Path("outputs")
    dataset = build_dataset(CONFIG["data"])
    model = build_model(CONFIG)
    result = train(model, dataset, CONFIG)

    plot_training_history(result, output_dir / "training_history.png")
    plot_checkpoint_embedding(result, method="pca", path=output_dir / "checkpoint_pca.png")
    inputs, targets = dataset.tensors
    plot_loss_landscape(
        model,
        inputs[:256],
        targets[:256],
        torch.nn.MSELoss(),
        radius=0.8,
        resolution=21,
        path=output_dir / "loss_landscape.png",
    )
    print(f"Final loss: {result.history['loss'][-1]:.6f}")
    print(f"Plots written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
