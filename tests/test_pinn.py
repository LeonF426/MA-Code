import math

import pytest
import torch
from torch import nn

from ssam import (
    build_model,
    build_poisson_points,
    constant_poisson_solution,
    evaluate_poisson_forcing_sensitivity,
    evaluate_poisson_laplace_sensitivity,
    evaluate_poisson_pinn,
    exact_poisson_solution,
    normalize_config,
    poisson_loss_components,
    poisson_residual,
    plot_pinn_training_history,
    plot_poisson_forcing_sensitivity,
    plot_poisson_solutions,
    poisson_forcing,
    train,
    train_poisson_pinn,
)


class ExactSolution(nn.Module):
    def forward(self, coordinates):
        return exact_poisson_solution(coordinates).unsqueeze(-1)


class TrainableExactSolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, coordinates):
        return (self.scale * exact_poisson_solution(coordinates)).unsqueeze(-1)


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


def test_poisson_residual_supports_scaled_forcing():
    points = build_poisson_points(_config())
    residual = poisson_residual(
        ExactSolution(),
        points.interior,
        create_graph=False,
        forcing_scale=1.5,
    )
    expected = -0.5 * 2.0 * math.pi**2 * exact_poisson_solution(points.interior)
    torch.testing.assert_close(residual, expected, atol=1e-9, rtol=1e-9)


def test_poisson_forcing_and_residual_support_additive_constant_uncertainty():
    points = build_poisson_points(_config())
    epsilon = 0.4
    forcing = poisson_forcing(
        points.interior,
        forcing_additive=epsilon,
    )
    nominal = poisson_forcing(points.interior)
    torch.testing.assert_close(
        forcing,
        nominal + epsilon * torch.ones_like(nominal),
    )

    residual = poisson_residual(
        ExactSolution(),
        points.interior,
        create_graph=False,
        forcing_additive=epsilon,
    )
    torch.testing.assert_close(
        residual,
        torch.full_like(residual, -epsilon),
        atol=1e-9,
        rtol=1e-9,
    )


def test_poisson_forcing_sensitivity_evaluates_frozen_model_once_per_sweep():
    config = _config("sgd")
    points = build_poisson_points(config)
    result = evaluate_poisson_forcing_sensitivity(
        TrainableExactSolution(),
        points,
        [0.5, 1.0, 1.5],
    )

    nominal = result.evaluations[1]
    increased = result.evaluations[2]
    assert nominal.nominal_normalized_l2_error == pytest.approx(0.0, abs=1e-12)
    assert nominal.pde_residual_rmse == pytest.approx(0.0, abs=1e-8)
    assert increased.nominal_normalized_l2_error == pytest.approx(0.5)
    assert increased.nominal_normalized_l2_error_change_from_nominal == pytest.approx(
        increased.nominal_normalized_l2_error
    )
    assert increased.rhs_sine_coefficient == pytest.approx(3.0 * math.pi**2)
    expected_rhs_change = float(
        (0.5 * 2.0 * math.pi**2 * exact_poisson_solution(points.evaluation))
        .square()
        .mean()
        .sqrt()
    )
    assert increased.rhs_change_rmse_from_nominal == pytest.approx(
        expected_rhs_change
    )
    assert result.worst_nominal_normalized_l2_error == pytest.approx(0.5)
    assert (
        result.max_nominal_normalized_l2_error_change_per_unit_scale
        == pytest.approx(1.0)
    )


def test_poisson_forcing_sensitivity_supports_additive_constant_family():
    config = _config("sgd")
    points = build_poisson_points(config)
    result = evaluate_poisson_forcing_sensitivity(
        TrainableExactSolution(),
        points,
        [-0.5, 0.0, 0.5],
        forcing_mode="additive",
        nominal_forcing_perturbation=0.0,
        additive_solution_terms=20,
    )

    nominal = result.evaluations[1]
    increased = result.evaluations[2]
    assert result.forcing_mode == "additive"
    assert result.nominal_forcing_perturbation == 0.0
    assert result.additive_solution_terms == 20
    assert nominal.nominal_normalized_l2_error == pytest.approx(0.0, abs=1e-12)
    assert nominal.pde_residual_rmse == pytest.approx(0.0, abs=1e-8)
    assert increased.forcing_scale == 1.0
    assert increased.forcing_perturbation == 0.5
    assert increased.rhs_constant == 0.5
    assert increased.rhs_change_rmse_from_nominal == pytest.approx(0.5)
    assert increased.pde_residual_rmse == pytest.approx(0.5, abs=1e-8)

    additive_solution = constant_poisson_solution(
        points.evaluation,
        series_terms=20,
    )
    nominal_solution = exact_poisson_solution(points.evaluation)
    expected_error = float(
        torch.linalg.vector_norm(0.5 * additive_solution)
        / torch.linalg.vector_norm(nominal_solution)
    )
    assert increased.nominal_normalized_l2_error == pytest.approx(expected_error)


def test_poisson_forcing_sensitivity_requires_included_nonzero_nominal_scale():
    config = _config("sgd")
    points = build_poisson_points(config)
    model = build_model(config).double()

    with pytest.raises(ValueError, match="include nominal"):
        evaluate_poisson_forcing_sensitivity(model, points, [0.5, 1.5])
    zero_scale_result = evaluate_poisson_forcing_sensitivity(
        model,
        points,
        [0.0, 1.0],
    )
    assert zero_scale_result.evaluations[0].forcing_scale == 0.0
    with pytest.raises(ValueError, match="nominal_forcing_scale must be non-zero"):
        evaluate_poisson_forcing_sensitivity(
            model,
            points,
            [0.0, 1.0],
            nominal_forcing_scale=0.0,
        )


def test_poisson_laplace_sensitivity_produces_local_metric_intervals():
    config = _config("sgd")
    points = build_poisson_points(config)
    model = TrainableExactSolution()
    model.train()
    result = evaluate_poisson_laplace_sensitivity(
        model,
        points,
        [0.5, 1.0, 1.5],
        pde_weight=2.0,
        boundary_weight=3.0,
        prior_precision=10.0,
        curvature_probes=4,
        seed=19,
    )

    assert model.training
    assert result.approximation == "diagonal_generalized_gauss_newton_laplace"
    assert result.parameter_count == 1
    assert 0.0 <= result.effective_parameter_count <= 1.0
    assert result.clean_gradient_norm == pytest.approx(0.0, abs=1e-8)
    for evaluation in result.evaluations:
        for metric in (
            evaluation.nominal_normalized_l2_error,
            evaluation.pde_residual_rmse,
        ):
            assert math.isfinite(metric.central)
            assert math.isfinite(metric.standard_deviation)
            assert metric.standard_deviation >= 0.0
            assert 0.0 <= metric.confidence_low <= metric.confidence_high


def test_poisson_laplace_sensitivity_supports_additive_constant_family():
    config = _config("sgd")
    points = build_poisson_points(config)
    result = evaluate_poisson_laplace_sensitivity(
        TrainableExactSolution(),
        points,
        [-0.25, 0.0, 0.25],
        prior_precision=10.0,
        curvature_probes=2,
        forcing_mode="additive",
        nominal_forcing_perturbation=0.0,
        additive_solution_terms=12,
    )

    assert result.forcing_mode == "additive"
    assert result.nominal_forcing_perturbation == 0.0
    assert result.additive_solution_terms == 12
    assert [value.forcing_perturbation for value in result.evaluations] == [
        -0.25,
        0.0,
        0.25,
    ]
    assert all(value.forcing_scale == 1.0 for value in result.evaluations)


def test_poisson_forcing_sensitivity_plot_shows_original_rhs_and_responses():
    config = _config("sgd")
    points = build_poisson_points(config)
    sensitivity = evaluate_poisson_forcing_sensitivity(
        TrainableExactSolution(),
        points,
        [0.5, 1.0, 1.5],
    )
    laplace = evaluate_poisson_laplace_sensitivity(
        TrainableExactSolution(),
        points,
        [0.5, 1.0, 1.5],
        prior_precision=10.0,
        curvature_probes=2,
    )

    figure = plot_poisson_forcing_sensitivity(
        {"exact": sensitivity},
        points,
        laplace_sensitivities={"exact": laplace},
    )
    titled_axes = {axis.get_title(): axis for axis in figure.axes if axis.get_title()}
    assert any("Original RHS" in title for title in titled_axes)
    assert len(titled_axes["Solution mismatch"].lines) == 2
    assert len(titled_axes["PDE residual mismatch"].lines) == 2
    assert titled_axes["Solution mismatch"].collections
    assert titled_axes["PDE residual mismatch"].collections


def test_poisson_forcing_sensitivity_plot_labels_additive_epsilon():
    config = _config("sgd")
    points = build_poisson_points(config)
    sensitivity = evaluate_poisson_forcing_sensitivity(
        TrainableExactSolution(),
        points,
        [-0.5, 0.0, 0.5],
        forcing_mode="additive",
        nominal_forcing_perturbation=0.0,
    )

    figure = plot_poisson_forcing_sensitivity(
        {"exact": sensitivity},
        points,
    )

    assert "additive shifts" in figure._suptitle.get_text()
    assert "epsilon" in figure.axes[1].get_xlabel()
    nominal_line = figure.axes[1].lines[-1]
    assert set(nominal_line.get_xdata()) == {0.0}


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


@pytest.mark.parametrize(
    ("sharpness_components", "expected_loss"),
    [
        (
            ["pde"],
            lambda history: (
                2.0 * history["gaussian_pde_loss"][0]
                + 3.0 * history["clean_boundary_loss"][0]
            ),
        ),
        (
            ["boundary"],
            lambda history: (
                2.0 * history["clean_pde_loss"][0]
                + 3.0 * history["gaussian_boundary_loss"][0]
            ),
        ),
    ],
)
def test_ssam_pinn_can_regularize_selected_loss_components(
    sharpness_components, expected_loss
):
    config = _config()
    config["training"]["steps"] = 1
    config["pinn"]["sharpness_components"] = sharpness_components
    result = train_poisson_pinn(
        build_model(config).double(),
        build_poisson_points(config),
        config,
    )

    assert result.history["regularized_loss"][0] == pytest.approx(
        expected_loss(result.history)
    )
    unselected_key = (
        "gaussian_boundary_loss"
        if sharpness_components == ["pde"]
        else "gaussian_pde_loss"
    )
    assert math.isnan(result.history[unselected_key][0])


def test_pinn_rejects_unknown_sharpness_component():
    config = _config()
    config["pinn"]["sharpness_components"] = ["initial"]

    with pytest.raises(ValueError, match="sharpness_components"):
        train_poisson_pinn(
            build_model(config).double(),
            build_poisson_points(config),
            config,
        )


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
        "inserted_lr": {"name": "constant", "value": 1e-3},
    }
    result = train_poisson_pinn(
        build_model(config).double(), build_poisson_points(config), config
    )
    assert 0.0 < result.history["learning_rate"][0] <= 1e-3


def test_poisson_pinn_uses_ratio_preserving_minibatches():
    config = _config("sgd")
    config["data"]["boundary_points"] = 4
    config["training"]["batch_size"] = 4
    config["training"]["steps"] = 1
    config["model"]["parameter_init"]["seed"] = 41

    result = train_poisson_pinn(
        build_model(config).double(),
        build_poisson_points(config),
        config,
    )

    assert result.config["effective_batch_sizes"] == {
        "interior": 4,
        "boundary": 2,
    }
    assert result.history["interior_batch_size"] == [4]
    assert result.history["boundary_batch_size"] == [2]


def test_poisson_batch_size_changes_the_parameter_update():
    full_config = _config("sgd")
    full_config["training"]["steps"] = 1
    full_config["model"]["parameter_init"]["seed"] = 73
    mini_config = normalize_config(full_config)
    mini_config["training"]["batch_size"] = 4
    points = build_poisson_points(full_config)

    full = train_poisson_pinn(build_model(full_config).double(), points, full_config)
    mini = train_poisson_pinn(build_model(mini_config).double(), points, mini_config)
    full_parameters = torch.cat([p.detach().reshape(-1) for p in full.model.parameters()])
    mini_parameters = torch.cat([p.detach().reshape(-1) for p in mini.model.parameters()])

    assert not torch.equal(full_parameters, mini_parameters)


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
    figure = plot_pinn_training_history({"sgd": history}, smoothing_window=1)
    assert len(figure.axes) == 3
    assert all(axis.get_yscale() == "log" for axis in figure.axes)
    assert all(len(axis.lines) == 1 for axis in figure.axes)


def test_pinn_history_plot_uses_distinct_green_for_gaussian_mean():
    history = {
        "step": [0, 1],
        "clean_loss": [2.0, 1.0],
        "regularized_loss": [2.2, 1.1],
        "clean_pde_loss": [1.5, 0.8],
        "gaussian_pde_loss": [1.7, 0.9],
        "clean_boundary_loss": [0.5, 0.2],
        "gaussian_boundary_loss": [0.55, 0.22],
    }

    figure = plot_pinn_training_history({"S-SAM": history}, smoothing_window=1)

    for axis in figure.axes:
        clean_line, gaussian_line = axis.lines
        assert gaussian_line.get_color().lower() == "#66c2a5"
        assert gaussian_line.get_color() != clean_line.get_color()


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
