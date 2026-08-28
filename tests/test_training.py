import torch

from ssam import build_model, make_linear_regression_dataset, train


def _model():
    return build_model({
        "name": "mlp",
        "input_dim": 2,
        "output_dim": 1,
        "depth": 1,
        "bias": False,
        "parameter_init": {"name": "zeros"},
    })


def _training(algorithm):
    return {
        "algorithm": algorithm,
        "steps": 30,
        "batch_size": 16,
        "learning_rate": {"name": "constant", "value": 0.05},
        "sharpness_scale": {"name": "linear", "start": 0.05, "end": 0.0, "duration": 30},
        "perturbation": {"samples": 2},
        "optimizer": {"name": "sgd"},
        "loss": "mse",
        "checkpoint_every": 5,
        "seed": 2,
    }


def test_gd_reduces_loss():
    data = make_linear_regression_dataset(64, 2, [2.0, -1.0], seed=1)
    result = train(_model(), data, _training("gd"))
    assert len(result.history["loss"]) == 30
    assert result.history["loss"][-1] < result.history["loss"][0]


def test_sgd_and_ssam_share_result_shape():
    data = make_linear_regression_dataset(64, 2, [2.0, -1.0], seed=1)
    sgd = train(_model(), data, _training("sgd"))
    ssam = train(_model(), data, _training("s_sam"))
    assert len(sgd.history["loss"]) == len(ssam.history["loss"]) == 30
    assert ssam.history["sharpness_scale"][0] == 0.05
    assert len(ssam.parameter_snapshots) >= 3
