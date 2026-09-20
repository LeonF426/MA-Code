"""Compare SGD and Gaussian S-SAM on a polynomial heat-equation PINN."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ssam import (
    build_heat_points,
    build_model,
    evaluate_average_sharpness_interpolation_closure,
    evaluate_heat_pinn,
    exact_heat_solution,
    heat_features,
    heat_loss_components,
    plot_pinn_training_history,
    plot_sharpness_interpolation,
    plot_space_time_pinn_solutions,
    train_heat_pinn,
)


CONFIG = {
    "model": {
        "name": "polynomial_heat_pinn",
        "type": "mixed_linear",
        "input_dim": 2,
        "layers": [
            {"type": "diag", "out_dim": 2, "activation": "identity"},
            {"type": "diag", "out_dim": 2, "activation": "identity"},
        ],
        "output_activation": "identity",
        "output_reduction": "sum",
        "bias": False,
        "parameter_init": {
            "type": "identity",
            "rescaling": {
                "mode": "layerwise",
                "log_scale_std": 0.5,
                "seed": 2320,
            },
        },
    },
    "data": {
        "name": "polynomial_heat_interval",
        "interior_points": 512,
        "initial_points": 256,
        "boundary_points": 256,
        "space_resolution": 51,
        "time_resolution": 51,
        "time_max": 1.0,
        "seed": 170,
    },
    "pinn": {
        "pde_weight": 1.0,
        "boundary_weight": 10.0,
        "algorithms": ["sgd", "s_sam"],
        "output_dir": "outputs/polynomial_heat_pinn",
        "evaluation": {
            "sharpness_scale": 1,
            "sharpness_samples": 64,
            "sharpness_seed": 90023,
            "antithetic": True,
            "interpolation_points": 11,
            "interpolation_samples": 64,
        },
    },
    "training": {
        "steps": 1000,
        "batch_size": 512,
        "learning_rate": {"name": "constant", "value": 1e-3},
        "sharpness_scale": {
            "name": "constant",
            "value": 0.1
        },
        "perturbation": {
            "distribution": "gaussian",
            "samples": 8,
            "normalized": False,
            "antithetic": True,
            "preserve_buffers": True,
        },
        "optimizer": {"name": "sgd", "momentum": 0.0},
        "seed": 2320,
        "device": "auto",
    },
}


def main() -> None:
    pinn_config = CONFIG["pinn"]
    evaluation_config = pinn_config["evaluation"]
    output_dir = Path(pinn_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = tuple(pinn_config.get("algorithms", ("sgd", "s_sam")))
    configs = {}
    for algorithm in algorithms:
        configs[algorithm] = copy.deepcopy(CONFIG)
        configs[algorithm]["training"]["algorithm"] = algorithm

    reference_config = configs[algorithms[0]]
    points = build_heat_points(reference_config)
    torch.manual_seed(int(CONFIG["training"]["seed"]))
    reference = build_model(reference_config)
    initial_state = copy.deepcopy(reference.state_dict())

    results = {}
    metrics = {}
    for algorithm, config in configs.items():
        model = build_model(config)
        model.load_state_dict(initial_state, strict=True)
        results[algorithm] = train_heat_pinn(model, points, config)
        metrics[algorithm] = evaluate_heat_pinn(
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

    for algorithm, values in metrics.items():
        sharpness = values.average_sharpness
        assert sharpness is not None
        print(
            f"{algorithm:>5} | rel L2={values.relative_l2_error:.4e} | "
            f"PDE RMSE={values.pde_residual_rmse:.4e} | "
            f"condition RMSE={values.boundary_rmse:.4e} | "
            f"sharpness={sharpness.average_sharpness:.4e} "
            f"+/- {1.96 * sharpness.standard_error:.2e}"
        )

    # interpolation = None
    if "sgd" in results and "s_sam" in results:
        interpolation_model = results["sgd"].model
        endpoint_model = results["s_sam"].model
        parameter = next(interpolation_model.parameters())
        interior = points.interior.to(device=parameter.device, dtype=parameter.dtype)
        initial = points.initial.to(device=parameter.device, dtype=parameter.dtype)
        boundary = points.boundary.to(device=parameter.device, dtype=parameter.dtype)

        def interpolation_loss():
            return heat_loss_components(
                interpolation_model,
                interior,
                initial,
                boundary,
                pde_weight=float(CONFIG["pinn"]["pde_weight"]),
                boundary_weight=float(CONFIG["pinn"]["boundary_weight"]),
            )["loss"]

        interpolation = evaluate_average_sharpness_interpolation_closure(
            interpolation_model,
            endpoint_model,
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
            normalized=bool(CONFIG["training"]["perturbation"]["normalized"]),
            requires_grad=True,
        )
        plot_sharpness_interpolation(
            interpolation,
            output_dir / "sharpness_interpolation.png",
            endpoint_labels=("SGD", "S-SAM"),
        )

    plot_space_time_pinn_solutions(
        {name: result.model for name, result in results.items()},
        points,
        heat_features,
        exact_heat_solution,
        output_dir / "solutions.png",
    )
    plot_pinn_training_history(results, output_dir / "training_history.png")
    (output_dir / "metrics.json").write_text(
        json.dumps({name: asdict(value) for name, value in metrics.items()}, indent=2),
        encoding="utf-8",
    )
    if interpolation is not None:
        (output_dir / "sharpness_interpolation.json").write_text(
            json.dumps(asdict(interpolation), indent=2),
            encoding="utf-8",
        )
    print(f"Plots and metrics written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
