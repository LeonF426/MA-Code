seed = 7

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
            "seed": seed,
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
            "inserted_lr": {
                "name": "constant",
                "value": 0.1,
            },
        },
        "sharpness_scale": {
            "name": "constant",
            "value": 0.05,
        },
        "perturbation": {
            "distribution": "gaussian",
            "samples": 8,
            "normalized": True,
            "antithetic": True,
            "seed": seed,
        },
        "optimizer": {
            "name": "sgd",
            "momentum": 0.0,
        },
        "loss": "mse",
        "checkpoint_every": 0,
        "seed": seed,
        "batch_seed": seed,
        "device": "auto",
    },
}


