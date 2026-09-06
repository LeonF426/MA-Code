"""The original mixed-linear experiment, now expressed as a general config case."""

from pathlib import Path
import copy

import torch
from torch.utils.data import DataLoader

from ssam import (
    build_dataset,
    build_model,
    plot_training_history,
    train,
    evaluate_average_sharpness
)

seed = 12

CONFIG_1 = {
    "model": {
        "name": "3L_2d_diag_linear_ssam",
        "type": "mixed_linear",
        "input_dim": 2,
        "layers": [
            {"type": "dense", "in_dim": 2, "out_dim": 2},
            {"type": "dense", "in_dim": 2, "out_dim": 2},
            {"type": "dense", "in_dim": 2, "out_dim": 2}
        ],
        "activation": "identity",
        "output_activation": "identity",
        "output_reduction": "sum",
        "bias": False,
        "parameter_init": {"type": "uniform", "low": -0.5, "high": 0.5},
    },
    "data": {
        "name": "linear_regression",
        "n_samples": 60,
        "input_dim": 2,
        "target_weights": [3.14159, -1.0],
        "noise_std": 0.05,
        "seed": seed ,
    },
    "training": {
        "algorithm": "s_sam",  # change to "gd" or "sgd"
        "steps": 500,
        "batch_size": 60,
        # "learning_rate": {"name": "constant", "value": 0.01},
        # "learning_rate": {
        #     "name": "strong_descent_diag",
        #     "delta": 0.5,
        #     "safety": 0.95,
        #     "max_lr": 0.1,
        #     "loss_floor": 1e-12,
        # },
        "learning_rate": {"name": "tamed",
                          "type": "sgd",
                          "inserted_lr":{
                          "name": "constant", "value": 0.1
        }
                          },
        #"sharpness_scale": {"name": "constant", "value": 1},
        "sharpness_scale": {"name": "inverse_time","initial": 2,"power": 0.25,"floor": 0.0,},
        "perturbation": {"distribution": "gaussian", "samples": 50},
        "optimizer": {"name": "sgd", "momentum": 0.0},
        "loss": "mse",
        "checkpoint_every": 1000,
        "seed": seed,
        "device": "auto",
    },
}

CONFIG_2 = copy.deepcopy(CONFIG_1)
CONFIG_2["training"]["algorithm"] = "sgd"

def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(CONFIG_1["data"])

    # Build the first model at the shared initial parameter estimate.
    torch.manual_seed(seed)
    model_1 = build_model(CONFIG_1)

    # deepcopy is essential because state_dict() tensors otherwise reference
    # the model's parameter storage and would change during training.
    initial_state = copy.deepcopy(model_1.state_dict())

    # Build the second model and overwrite its initialization with the exact
    # same parameters and buffers used by model_1.
    model_2 = build_model(CONFIG_2)
    model_2.load_state_dict(initial_state, strict=True)

    # Verify equality before either model is trained.
    for parameter_1, parameter_2 in zip(
        model_1.parameters(),
        model_2.parameters(),
    ):
        torch.testing.assert_close(parameter_1, parameter_2)

    # The two separate model objects now start identically.
    result_1 = train(model_1, dataset, CONFIG_1)
    result_2 = train(model_2, dataset, CONFIG_2)

    assert result_1.model is model_1
    assert result_2.model is model_2

    # ---------------------------------------------------------------
    # Average-sharpness evaluation goes here, after training.
    # ---------------------------------------------------------------

    evaluation_loader = DataLoader(
        dataset,
        batch_size=len(dataset),  # Full dataset; use less if memory requires it.
        shuffle=False,
        drop_last=False,
    )

    evaluation_loss = torch.nn.MSELoss()

    # This is the radius at which sharpness is compared. It should be identical
    # for every trained model and does not need to equal the final training eta.
    evaluation_scale = 3

    # Use many perturbations for an accurate estimate.
    evaluation_samples = 4096
    evaluation_seed = 12345

    sharpness_1 = evaluate_average_sharpness(
        result_1.model,  # Equivalent to using model_1.
        evaluation_loader,
        evaluation_loss,
        sharpness_scale=evaluation_scale,
        samples=evaluation_samples,
        seed=evaluation_seed,
        antithetic=True,
    )

    sharpness_2 = evaluate_average_sharpness(
        result_2.model,  # Equivalent to using model_2.
        evaluation_loader,
        evaluation_loss,
        sharpness_scale=evaluation_scale,
        samples=evaluation_samples,
        seed=evaluation_seed,
        antithetic=True,
    )

    print(f"Final S-SAM loss: {result_1.history['loss'][-1]:.6f}")
    print(f"Final GD loss:    {result_2.history['loss'][-1]:.6f}")

    print()
    print(
        "S-SAM average sharpness: "
        f"{sharpness_1.average_sharpness:.8g} "
        f"± {1.96 * sharpness_1.standard_error:.3g}"
    )
    print(
        "GD average sharpness:    "
        f"{sharpness_2.average_sharpness:.8g} "
        f"± {1.96 * sharpness_2.standard_error:.3g}"
    )

    plot_training_history(
        result_1,
        output_dir
        / f"{CONFIG_1['model']['name']}_{CONFIG_1['training']['algorithm']}.png",
    )

    plot_training_history(
        result_2,
        output_dir
        / f"{CONFIG_2['model']['name']}_{CONFIG_2['training']['algorithm']}.png",
    )


if __name__ == "__main__":
    main()

