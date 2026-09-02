seed = None

CONFIG = {
    "model": {
        "name": "3L_2d_linear_inv_eta",
        "type": "mlp",
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
        "algorithm": "s_sam",  # change to "gd" or "sgd"
        "steps": 500,
        "batch_size": 40,
        #"learning_rate": {"name": "constant", "value": 0.01},
        "learning_rate": {
            "name": "strong_descent_diag",
            "delta": 0.5,
            "safety": 0.95,
            "max_lr": 0.1,
            "loss_floor": 1e-12,
        },
        "learning_rate": {"name": "tamed",
                          "type": "sgd",
                          "inserted_lr":{
            "name": "strong_descent_diag",
            "delta": 0.5,
            "safety": 0.95,
            "max_lr": 0.1,
            "loss_floor": 1e-12,
        }
                          },
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
