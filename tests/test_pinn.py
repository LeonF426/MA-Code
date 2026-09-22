import math

import pytest
import torch
from torch import nn

from ssam import (
    build_model,
    build_poisson_points,
    evaluate_poisson_pinn,
    exact_poisson_solution,
    normalize_config,
    poisson_loss_components,
    poisson_residual,
    plot_pinn_training_history,
    plot_poisson_solutions,
    train,
    train_poisson_pinn,
)


class ExactSolution(nn.Module):
    def forward(self, coordinates):
        return exact_poisson_solution(coordinates).unsqueeze(-1)


def _config(algorithm="s_sam"):
    return {
        "model": {
            "name": "tiny_poisson",
            "type": "mlp",
            "input_dim": 2,
            "output_dim": 1,
            "depth": 2,
            "width": 6,
            "activation": "tanh",
            "parameter_init": {"type": "xavier_uniform"},
        },
        "data": {
            "interior_points": 8,
            "boundary_points": 8,
            "evaluation_resolution": 5,
            "seed": 4,
            "dtype": "float64",
        },
        "pinn": {"pde_weight": 2.0, "boundary_weight": 3.0},
        "training": {
            "algorithm": algorithm,
            "steps": 2,
            "batch_size": 8,
            "learning_rate": {"name": "constant", "value": 1e-3},
            "sharpness_scale": {"name": "constant", "value": 1e-4},
            "perturbation": {"samples": 2, "antithetic": True},
            "optimizer": {"name": "sgd"},
            "device": "cpu",
            "seed": 5,
        },
    }


def test_exact_solution_has_zero_poisson_residual_and_boundary_loss():
    points = build_poisson_points(_config())
    residual = poisson_residual(ExactSolution(), points.interior, create_graph=False)
    losses = poisson_loss_components(
        ExactSolution(), points.interior, points.boundary
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-9, rtol=1e-9)
    assert float(losses["boundary_loss"]) == pytest.approx(0.0, abs=1e-30)


def test_ssam_pinn_tracks_clean_and_gaussian_component_losses():
    config = _config()
    model = build_model(config).double()
    points = build_poisson_points(config)
    result = train_poisson_pinn(model, points, config)

    for key in (
        "clean_pde_loss",
        "clean_boundary_loss",
        "gaussian_pde_loss",
        "gaussian_boundary_loss",
        "regularized_loss",
    ):
        assert len(result.history[key]) == 2
        assert all(math.isfinite(value) for value in result.history[key])
    expected = (
        2.0 * result.history["clean_pde_loss"][0]
        + 3.0 * result.history["clean_boundary_loss"][0]
    )
    assert result.history["clean_loss"][0] == pytest.approx(expected)


def test_pinn_sharpness_evaluation_supports_coordinate_derivatives():
    config = _config("sgd")
    model = build_model(config).double()
    points = build_poisson_points(config)
    metrics = evaluate_poisson_pinn(
        model,
        points,
        sharpness_scale=0.0,
        sharpness_samples=4,
        sharpness_seed=12,
    )
    assert math.isfinite(metrics.relative_l2_error)
    assert math.isfinite(metrics.pde_residual_rmse)
    assert metrics.average_sharpness is not None
    assert metrics.average_sharpness.average_sharpness == pytest.approx(0.0)


def test_poisson_pinn_supports_tamed_learning_rate():
    config = _config()
    config["training"]["steps"] = 1
    config["training"]["learning_rate"] = {
        "name": "tamed",
        "type": "sgd",
        "inserted_lr": {"name": "constant", "value": 1e-3},
    }
    result = train_poisson_pinn(
        build_model(config).double(), build_poisson_points(config), config
    )
    assert 0.0 < result.history["learning_rate"][0] <= 1e-3


def test_supervised_ssam_keeps_no_grad_clean_loss_default():
    config = {
        "model": {"name": "mlp", "input_dim": 1, "output_dim": 1, "depth": 1},
        "training": {
            "algorithm": "s_sam",
            "steps": 1,
            "batch_size": 2,
            "learning_rate": {"name": "constant", "value": 0.01},
            "sharpness_scale": {"name": "constant", "value": 0.01},
            "perturbation": {"samples": 1},
            "device": "cpu",
        },
    }
    grad_modes = []

    def observed_mse(predictions, targets):
        grad_modes.append(torch.is_grad_enabled())
        return nn.functional.mse_loss(predictions, targets)

    data = (torch.ones(2, 1), torch.zeros(2, 1))
    train(build_model(config), data, config, loss_fn=observed_mse)
    assert normalize_config(config)["training"]["loss_requires_grad"] is False
    assert grad_modes == [False, True]


def test_pinn_history_plot_skips_unavailable_gaussian_curves():
    history = {
        "step": [0, 1],
        "clean_loss": [2.0, 1.0],
        "clean_pde_loss": [1.5, 0.8],
        "clean_boundary_loss": [0.5, 0.2],
    }
    figure = plot_pinn_training_history({"sgd": history})
    assert len(figure.axes) == 3


def test_poisson_solution_and_prediction_panels_share_color_scale():
    config = _config("sgd")
    points = build_poisson_points(config)
    first = build_model(config).double()
    second = build_model(config).double()
    with torch.no_grad():
        for parameter in second.parameters():
            parameter.mul_(3.0)

    figure = plot_poisson_solutions({"first": first, "second": second}, points)
    solution_axes = [
        axis
        for axis in figure.axes
        if axis.images and "error" not in axis.get_title().lower()
    ]
    color_limits = [axis.images[0].get_clim() for axis in solution_axes]

    assert len(color_limits) == 4
    assert all(limits == color_limits[0] for limits in color_limits[1:])
