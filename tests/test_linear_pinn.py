import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from ssam import (
    advection_loss_components,
    advection_residual,
    build_advection_points,
    build_heat_points,
    build_model,
    build_polynomial_poisson_points,
    evaluate_average_sharpness_interpolation_closure,
    evaluate_advection_pinn,
    evaluate_heat_pinn,
    evaluate_polynomial_poisson_pinn,
    heat_loss_components,
    heat_residual,
    polynomial_poisson_loss_components,
    polynomial_poisson_residual,
    train_advection_pinn,
    train_heat_pinn,
    train_polynomial_poisson_pinn,
)


class FixedFeatureCoefficients(nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        self.register_buffer(
            "coefficients", torch.tensor(coefficients, dtype=torch.float64)
        )

    def forward(self, features):
        return features @ self.coefficients


def _load_example(name):
    path = Path(__file__).parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(dimension, algorithm="s_sam"):
    return {
        "model": {
            "name": "tiny_linear_pinn",
            "type": "mixed_linear",
            "input_dim": dimension,
            "layers": [
                {"type": "diag", "out_dim": dimension, "activation": "identity"},
                {"type": "diag", "out_dim": dimension},
            ],
            "output_reduction": "sum",
            "bias": False,
            "parameter_init": {"type": "identity"},
        },
        "data": {
            "interior_points": 8,
            "initial_points": 6,
            "boundary_points": 8,
            "space_resolution": 5,
            "time_resolution": 5,
            "evaluation_resolution": 7,
            "dtype": "float64",
            "seed": 4,
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


def test_exact_affine_advection_has_zero_loss():
    points = build_advection_points(_config(3))
    model = FixedFeatureCoefficients([2.0, 3.0, 3.0])
    residual = advection_residual(model, points.interior, create_graph=False)
    losses = advection_loss_components(
        model, points.interior, points.initial, points.boundary
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert float(losses["loss"].detach()) == pytest.approx(0.0, abs=1e-30)


def test_exact_polynomial_poisson_has_zero_loss():
    points = build_polynomial_poisson_points(_config(3))
    model = FixedFeatureCoefficients([1.0, 3.0, 2.0])
    residual = polynomial_poisson_residual(
        model, points.interior, create_graph=False
    )
    losses = polynomial_poisson_loss_components(
        model, points.interior, points.boundary
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert float(losses["loss"].detach()) == pytest.approx(0.0, abs=1e-30)


def test_exact_polynomial_heat_solution_has_zero_loss():
    points = build_heat_points(_config(2))
    model = FixedFeatureCoefficients([1.0, 2.0])
    residual = heat_residual(model, points.interior, create_graph=False)
    losses = heat_loss_components(
        model, points.interior, points.initial, points.boundary
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert float(losses["loss"].detach()) == pytest.approx(0.0, abs=1e-30)


@pytest.mark.parametrize(
    ("example_name", "dimension"),
    [
        ("advection_pinn", 3),
        ("polynomial_heat_pinn", 2),
        ("polynomial_poisson_pinn", 3),
    ],
)
def test_linear_pinn_examples_start_functionally_equivalent_but_imbalanced(
    example_name, dimension
):
    config = _load_example(example_name).CONFIG
    model = build_model(config)
    inputs = torch.randn(7, dimension)

    # Rescaling changes the factorization, not the identity-initialized function.
    torch.testing.assert_close(model(inputs), inputs.sum(dim=-1))
    assert model.rescaling_result.mode == "layerwise"
    assert model.rescaling_result.log_scale_std == pytest.approx(0.5)

    layer_energies = torch.tensor(
        [float(layer.weight.detach().square().sum()) for layer in model.layers]
    )
    assert float(layer_energies.max() - layer_energies.min()) > 1e-3


@pytest.mark.parametrize(
    ("dimension", "build_points", "train_pinn", "evaluate_pinn"),
    [
        (3, build_advection_points, train_advection_pinn, evaluate_advection_pinn),
        (
            3,
            build_polynomial_poisson_points,
            train_polynomial_poisson_pinn,
            evaluate_polynomial_poisson_pinn,
        ),
        (2, build_heat_points, train_heat_pinn, evaluate_heat_pinn),
    ],
)
def test_linear_pinn_ssam_training_and_evaluation(
    dimension, build_points, train_pinn, evaluate_pinn
):
    config = _config(dimension)
    points = build_points(config)
    model = build_model(config).double()
    result = train_pinn(model, points, config)
    metrics = evaluate_pinn(
        model,
        points,
        sharpness_scale=0.0,
        sharpness_samples=2,
    )

    assert len(result.history["step"]) == 2
    assert all(math.isfinite(value) for value in result.history["clean_loss"])
    assert math.isfinite(metrics.relative_l2_error)
    assert math.isfinite(metrics.pde_residual_rmse)
    assert metrics.average_sharpness is not None
    assert metrics.average_sharpness.average_sharpness == pytest.approx(0.0)


def test_interpolation_sharpness_supports_coordinate_derivatives():
    config = _config(3, "sgd")
    points = build_polynomial_poisson_points(config)
    start = build_model(config).double()
    end = build_model(config).double()
    with torch.no_grad():
        for parameter in end.parameters():
            parameter.add_(0.1)

    parameter = next(start.parameters())
    interior = points.interior.to(device=parameter.device, dtype=parameter.dtype)
    boundary = points.boundary.to(device=parameter.device, dtype=parameter.dtype)

    def closure():
        return polynomial_poisson_loss_components(
            start,
            interior,
            boundary,
            pde_weight=2.0,
            boundary_weight=3.0,
        )["loss"]

    result = evaluate_average_sharpness_interpolation_closure(
        start,
        end,
        closure,
        0.0,
        interpolation_points=3,
        samples=2,
        requires_grad=True,
    )

    assert len(result.points) == 3
    assert all(math.isfinite(point.clean_loss) for point in result.points)
    assert all(point.average_sharpness == pytest.approx(0.0) for point in result.points)
