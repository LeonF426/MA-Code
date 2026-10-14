"""Compare SGD and Gaussian S-SAM on a 2D Poisson PINN."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
import torch

from ssam import (
    build_model,
    build_poisson_points,
    create_run_artifacts,
    evaluate_average_sharpness_closure,
    evaluate_average_sharpness_interpolation_closure,
    evaluate_poisson_forcing_sensitivity,
    evaluate_poisson_laplace_sensitivity,
    evaluate_poisson_pinn,
    plot_pinn_training_history,
    plot_poisson_forcing_sensitivity,
    plot_poisson_solutions,
    plot_sharpness_interpolation,
    poisson_loss_components,
    train_poisson_pinn,
    write_training_history,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 22_478_666


def build_config(seed: int = SEED) -> dict[str, Any]:
    """Build the experiment config from one base seed and fixed offsets."""

    seed = int(seed)
    return {
        # Optional human-readable prefix for the timestamped run directory.
        "run": {"name": None},
        "model": {
            "name": "poisson_pinn",
            "type": "mlp",
            "input_dim": 2,
            "output_dim": 1,
            "depth": 4,
            "width": 64,
            "activation": "tanh",
            "output_activation": "identity",
            "bias": True,
            "parameter_init": {
                "type": "xavier_uniform",
                "rescaling": {
                    # Tanh is not positively homogeneous, so this intentionally
                    # changes the initial function while altering parameter scales.
                    "preserve_function": False,
                    "mode": "layerwise",
                    "log_scale_std": 0.04,
                    "seed": seed + 1,
                },
                "seed": seed + 2,
            },
        },
        "data": {
            "name": "poisson_unit_square",
            "interior_points": 512,
            "boundary_points": 256,
            "evaluation_resolution": 51,
            "seed": seed + 3,
        },
        "pinn": {
            "pde_weight": 1.0,
            "boundary_weight": 7.5,
            "sharpness_components": ["pde", "boundary"],
            "algorithms": ["sgd", "s_sam"],
            "output_dir": "outputs/poisson_pinn",
            "evaluation": {
                "sharpness_scale": 1e-1,
                "sharpness_samples": 100,
                "sharpness_seed": seed + 4,
                "antithetic": True,
                "interpolation_points": 11,
                "interpolation_samples": 32,
                # Choose "scaled" for a*f or "additive" for f + epsilon*g with
                # g=1. Both value grids remain in the config, so switching only
                # requires changing this one mode.
                "forcing_uncertainty": {
                    "mode": "scaled",
                    "scaled": {
                        "values": [
                            0.95,
                            0.96,
                            0.97,
                            0.98,
                            0.99,
                            1.0,
                            1.01,
                            1.02,
                            1.03,
                            1.04,
                            1.05,
                        ],
                        "nominal": 1.0,
                    },
                    "additive": {
                        "values": [
                            -0.1,
                            -0.075,
                            -0.05,
                            -0.025,
                            0.0,
                            0.025,
                            0.05,
                            0.075,
                            0.1,
                        ],
                        "nominal": 0.0,
                        "g": 1.0,
                        "solution_series_terms": 50,
                    },
                },
                # Local epistemic-uncertainty approximation around each final
                # parameter vector. These bands do not randomize the RHS.
                "laplace": {
                    "enabled": False,
                    "prior_precision": 100.0,
                    "likelihood_precision": 1.0,
                    "curvature_probes": 16,
                    "confidence_level": 0.95,
                    "seed": seed + 5,
                },
            },
        },
        "training": {
            "steps": 1200,
            "batch_size": 128,
            "learning_rate": {
                "name": "tamed",
                "inserted_lr": {"name": "constant", "value": 2e-3},
            },
            "sharpness_scale": {
                "name": "inverse_time",
                "initial": 1,
                "power": 0.5,
                "floor": 0.0,
            },
            # "sharpness_scale": {
            #     "name": "constant",
            #     "value": 0.05
            # },
            "perturbation": {
                "distribution": "gaussian",
                "samples": 20,
                "normalized": False,
                "antithetic": True,
                "preserve_buffers": True,
                "seed": seed + 6,
            },
            "optimizer": {"name": "sgd", "momentum": 0.0},
            "seed": seed + 7,
            "batch_seed": seed + 8,
            "device": "auto",
        },
        "visualization": {
            "training_history": {"smoothing_window": 10, "loss_start_step": 10},
        },
    }


CONFIG = build_config()


def main(config: Mapping[str, Any] | None = None) -> Path:
    configuration = copy.deepcopy(CONFIG if config is None else config)
    pinn_config = configuration["pinn"]
    evaluation_config = pinn_config["evaluation"]
    forcing_config = evaluation_config["forcing_uncertainty"]
    forcing_mode = str(forcing_config["mode"]).lower()
    if forcing_mode not in {"scaled", "additive"}:
        raise ValueError(
            "pinn.evaluation.forcing_uncertainty.mode must be "
            "'scaled' or 'additive'"
        )
    active_forcing_config = forcing_config[forcing_mode]
    if (
        forcing_mode == "additive"
        and float(active_forcing_config.get("g", 1.0)) != 1.0
    ):
        raise ValueError("The additive Poisson RHS currently requires g=1")
    forcing_values = active_forcing_config["values"]
    nominal_forcing_perturbation = float(active_forcing_config["nominal"])
    additive_solution_terms = int(
        active_forcing_config.get("solution_series_terms", 50)
    )
    algorithms = tuple(pinn_config.get("algorithms", ("sgd", "s_sam")))
    configs = {}
    for algorithm in algorithms:
        configs[algorithm] = copy.deepcopy(configuration)
        configs[algorithm]["training"]["algorithm"] = algorithm
    run = create_run_artifacts(
        {"run": copy.deepcopy(configuration["run"]), "algorithms": configs},
        PROJECT_ROOT / pinn_config["output_dir"],
        "poisson_pinn",
    )
    output_dir = run.output_dir

    # One point object is shared by training, metrics, and both algorithms.
    reference_config = configs[algorithms[0]]
    points = build_poisson_points(reference_config)
    reference = build_model(reference_config)
    initial_state = copy.deepcopy(reference.state_dict())

    def evaluate_initial_sharpness(model):
        parameter = next(model.parameters())
        interior = points.interior.to(device=parameter.device, dtype=parameter.dtype)
        boundary = points.boundary.to(device=parameter.device, dtype=parameter.dtype)

        def loss_closure():
            return poisson_loss_components(
                model,
                interior,
                boundary,
                pde_weight=float(configuration["pinn"]["pde_weight"]),
                boundary_weight=float(configuration["pinn"]["boundary_weight"]),
            )["loss"]

        measurement = evaluate_average_sharpness_closure(
            model,
            loss_closure,
            float(evaluation_config["sharpness_scale"]),
            samples=int(evaluation_config["sharpness_samples"]),
            seed=int(evaluation_config["sharpness_seed"]),
            antithetic=bool(evaluation_config.get("antithetic", True)),
            normalized=bool(
                configuration["training"]["perturbation"]["normalized"]
            ),
            requires_grad=True,
        )
        print(
            "initial | average sharpness"
            f"(scale={measurement.sharpness_scale:.8g})="
            f"{measurement.average_sharpness:.8e}"
        )
        return measurement

    def evaluate_interpolation(sgd_model, ssam_model):
        interpolation_model = sgd_model
        parameter = next(interpolation_model.parameters())
        interior = points.interior.to(device=parameter.device, dtype=parameter.dtype)
        boundary = points.boundary.to(device=parameter.device, dtype=parameter.dtype)

        def interpolation_loss():
            return poisson_loss_components(
                interpolation_model,
                interior,
                boundary,
                pde_weight=float(configuration["pinn"]["pde_weight"]),
                boundary_weight=float(configuration["pinn"]["boundary_weight"]),
            )["loss"]

        interpolation = evaluate_average_sharpness_interpolation_closure(
            interpolation_model,
            ssam_model,
            interpolation_loss,
            float(evaluation_config["sharpness_scale"]),
            interpolation_points=int(evaluation_config.get("interpolation_points", 11)),
            samples=int(
                evaluation_config.get(
                    "interpolation_samples",
                    evaluation_config["sharpness_samples"],
                )
            ),
            seed=int(evaluation_config["sharpness_seed"]),
            antithetic=bool(evaluation_config.get("antithetic", True)),
            normalized=bool(
                configuration["training"]["perturbation"]["normalized"]
            ),
            requires_grad=True,
        )
        plot_sharpness_interpolation(
            interpolation,
            output_dir / "sharpness_interpolation.png",
            endpoint_labels=("SGD", "S-SAM"),
        )
        (output_dir / "sharpness_interpolation.json").write_text(
            json.dumps(asdict(interpolation), indent=2),
            encoding="utf-8",
        )
        return interpolation

    models = {}
    for algorithm, config in configs.items():
        model = build_model(config)
        model.load_state_dict(initial_state, strict=True)
        models[algorithm] = model

    initial_sharpness = evaluate_initial_sharpness(models[algorithms[0]])

    results = {}
    metrics = {}
    forcing_sensitivities = {}
    laplace_sensitivities = {}
    laplace_config = evaluation_config.get("laplace", {})
    for algorithm, config in configs.items():
        model = models[algorithm]
        results[algorithm] = train_poisson_pinn(model, points, config)
        metrics[algorithm] = evaluate_poisson_pinn(
            model,
            points,
            pde_weight=config["pinn"]["pde_weight"],
            boundary_weight=config["pinn"]["boundary_weight"],
            sharpness_scale=float(evaluation_config["sharpness_scale"]),
            sharpness_samples=int(evaluation_config["sharpness_samples"]),
            sharpness_seed=int(evaluation_config["sharpness_seed"]),
            antithetic=bool(evaluation_config.get("antithetic", True)),
            normalized=config["training"]["perturbation"]["normalized"],
        )
        forcing_sensitivities[algorithm] = evaluate_poisson_forcing_sensitivity(
            model,
            points,
            forcing_values,
            forcing_mode=forcing_mode,
            nominal_forcing_perturbation=nominal_forcing_perturbation,
            additive_solution_terms=additive_solution_terms,
        )
        if bool(laplace_config.get("enabled", True)):
            laplace_sensitivities[algorithm] = evaluate_poisson_laplace_sensitivity(
                model,
                points,
                forcing_values,
                pde_weight=float(config["pinn"]["pde_weight"]),
                boundary_weight=float(config["pinn"]["boundary_weight"]),
                prior_precision=float(laplace_config.get("prior_precision", 100.0)),
                likelihood_precision=float(
                    laplace_config.get("likelihood_precision", 1.0)
                ),
                curvature_probes=int(laplace_config.get("curvature_probes", 16)),
                confidence_level=float(
                    laplace_config.get("confidence_level", 0.95)
                ),
                seed=int(laplace_config.get("seed", 2718)),
                forcing_mode=forcing_mode,
                nominal_forcing_perturbation=nominal_forcing_perturbation,
                additive_solution_terms=additive_solution_terms,
            )

    original_rhs = {
        "forcing_scale": 1.0,
        "equation": "-Delta u = f(x,y)",
        "expression": "f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)",
        "rhs_sine_coefficient": 2.0 * math.pi**2,
    }
    print(
        "ORIGINAL RHS | -Delta u = f(x,y) = "
        f"{original_rhs['rhs_sine_coefficient']:.6g} "
        "sin(pi*x) sin(pi*y)"
    )
    forcing_label = "scale" if forcing_mode == "scaled" else "epsilon"

    for algorithm, values in metrics.items():
        sharpness = values.average_sharpness
        assert sharpness is not None
        print(
            f"{algorithm:>5} | rel L2={values.relative_l2_error:.4e} | "
            f"PDE RMSE={values.pde_residual_rmse:.4e} | "
            f"boundary RMSE={values.boundary_rmse:.4e} | "
            f"sharpness(scale={sharpness.sharpness_scale:.8g})="
            f"{sharpness.average_sharpness:.4e} "
            f"+/- {1.96 * sharpness.standard_error:.2e}"
        )
        sensitivity = forcing_sensitivities[algorithm]
        for evaluation in sensitivity.evaluations:
            rhs_kind = (
                "ORIGINAL"
                if math.isclose(
                    evaluation.forcing_perturbation,
                    nominal_forcing_perturbation,
                )
                else "PERTURBED"
            )
            additive_rhs = (
                ""
                if forcing_mode == "scaled"
                else f" {evaluation.rhs_constant:+.6g}"
            )
            print(
                f"{algorithm:>5} | {rhs_kind} RHS "
                f"{forcing_label}={evaluation.forcing_perturbation:g}: "
                f"-Delta u = {evaluation.rhs_sine_coefficient:.6g} "
                f"sin(pi*x) sin(pi*y){additive_rhs} | "
                f"RHS RMSE={evaluation.rhs_rmse:.4e} | "
                f"RHS change RMSE={evaluation.rhs_change_rmse_from_nominal:.4e} | "
                "nominal-normalized L2="
                f"{evaluation.nominal_normalized_l2_error:.4e} | "
                f"PDE RMSE={evaluation.pde_residual_rmse:.4e}"
            )
        print(
            f"{algorithm:>5} | RHS sensitivity | "
            "worst nominal-normalized L2="
            f"{sensitivity.worst_nominal_normalized_l2_error:.4e} | "
            f"worst PDE RMSE={sensitivity.worst_pde_residual_rmse:.4e} | "
            f"max changes/unit {forcing_label}: "
            "nominal-normalized L2="
            f"{sensitivity.max_nominal_normalized_l2_error_change_per_unit_scale:.4e}, "
            "PDE RMSE="
            f"{sensitivity.max_pde_residual_rmse_change_per_unit_scale:.4e}"
        )
        laplace = laplace_sensitivities.get(algorithm)
        if laplace is not None:
            print(
                f"{algorithm:>5} | Laplace ({laplace.approximation}) | "
                f"clean gradient norm={laplace.clean_gradient_norm:.4e} | "
                f"effective parameters={laplace.effective_parameter_count:.2f}/"
                f"{laplace.parameter_count} | posterior parameter std: "
                f"mean={laplace.posterior_parameter_std_mean:.4e}, "
                f"max={laplace.posterior_parameter_std_max:.4e}"
            )
            for evaluation in laplace.evaluations:
                solution_error = evaluation.nominal_normalized_l2_error
                residual = evaluation.pde_residual_rmse
                confidence_percent = 100.0 * laplace.confidence_level
                print(
                    f"{algorithm:>5} | Laplace RHS {forcing_label}="
                    f"{evaluation.forcing_perturbation:g} | "
                    "nominal-normalized L2="
                    f"{solution_error.central:.4e} "
                    f"[{solution_error.confidence_low:.4e}, "
                    f"{solution_error.confidence_high:.4e}] | "
                    f"PDE RMSE={residual.central:.4e} "
                    f"[{residual.confidence_low:.4e}, "
                    f"{residual.confidence_high:.4e}] "
                    f"({confidence_percent:g}% local interval)"
                )

    if "sgd" in results and "s_sam" in results:
        evaluate_interpolation(
            results["sgd"].model,
            results["s_sam"].model,
        )

    plot_poisson_solutions(
        {name: result.model for name, result in results.items()},
        points,
        output_dir / "solutions.png",
    )
    plot_poisson_forcing_sensitivity(
        forcing_sensitivities,
        points,
        output_dir / "forcing_sensitivity.png",
        laplace_sensitivities=laplace_sensitivities,
    )
    history_plot = configuration["visualization"]["training_history"]
    plot_pinn_training_history(
        results,
        output_dir / "training_history.png",
        smoothing_window=int(history_plot["smoothing_window"]),
        loss_start_step=int(history_plot["loss_start_step"]),
    )
    write_training_history(results, output_dir / "training_history.json")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "original_rhs": original_rhs,
                "uncertainty_interpretation": {
                    "forcing_sensitivity": (
                        "Deterministic frozen-model stress test under a hand-selected "
                        f"{forcing_mode} RHS family; not a probability model."
                    ),
                    "laplace_sensitivity": (
                        "Local diagonal-GGN Gaussian parameter approximation with "
                        "linearized metric propagation; intervals depend on the "
                        "configured prior and likelihood precisions."
                    ),
                },
                "initial_average_sharpness": asdict(initial_sharpness),
                "training": {
                    name: result.config for name, result in results.items()
                },
                "algorithms": {
                    name: {
                        **asdict(value),
                        "forcing_sensitivity": asdict(
                            forcing_sensitivities[name]
                        ),
                        "laplace_sensitivity": (
                            asdict(laplace_sensitivities[name])
                            if name in laplace_sensitivities
                            else None
                        ),
                    }
                    for name, value in metrics.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    run.complete()
    print(f"Plots and metrics written to {output_dir.resolve()}")
    return output_dir


if __name__ == "__main__":
    main()
