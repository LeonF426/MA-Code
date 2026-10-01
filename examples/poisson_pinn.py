"""Compare SGD and Gaussian S-SAM on a 2D Poisson PINN."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
import torch

from ssam import (
    build_model,
    build_poisson_points,
    create_run_artifacts,
    evaluate_average_sharpness_closure,
    evaluate_average_sharpness_interpolation_closure,
    evaluate_poisson_pinn,
    plot_pinn_training_history,
    plot_poisson_solutions,
    plot_sharpness_interpolation,
    poisson_loss_components,
    train_poisson_pinn,
)


CONFIG = {
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
                "seed": 272442342,
            },
            "seed": 123939
        },
    },
    "data": {
        "name": "poisson_unit_square",
        "interior_points": 512,
        "boundary_points": 256,
        "evaluation_resolution": 51,
        "seed": 1753,
    },
    "pinn": {
        "pde_weight": 1.0,
        "boundary_weight": 7.5,
        "sharpness_components": ["pde","boundary"],
        "algorithms": ["sgd", "s_sam"],
        "output_dir": "outputs/poisson_pinn",
        "evaluation": {
            "sharpness_scale": 1e-1,
            "sharpness_samples": 100,
            "sharpness_seed": 1023,
            "antithetic": True,
            "interpolation_points": 11,
            "interpolation_samples": 32,
        },
    },
    "training": {
        "steps": 1200,
        "batch_size": 128,
        "learning_rate": {"name": "tamed",
                          "type": "sgd",
                          "inserted_lr":{"name": "constant", "value": 2e-3}},
        "sharpness_scale": {
            "name": "inverse_time",
            "initial": 1,
            "power": 0.5,
            "floor": 0.0,
        },
        "perturbation": {
            "distribution": "gaussian",
            "samples": 20,
            "normalized": False,
            "antithetic": True,
            "preserve_buffers": True,
        },
        "optimizer": {"name": "sgd", "momentum": 0.0},
        "seed": 232413,
        "device": "auto",
    },
}


def main() -> None:
    pinn_config = CONFIG["pinn"]
    evaluation_config = pinn_config["evaluation"]
    algorithms = tuple(pinn_config.get("algorithms", ("sgd", "s_sam")))
    configs = {}
    for algorithm in algorithms:
        configs[algorithm] = copy.deepcopy(CONFIG)
        configs[algorithm]["training"]["algorithm"] = algorithm
    run = create_run_artifacts(
        {"run": copy.deepcopy(CONFIG["run"]), "algorithms": configs},
        pinn_config["output_dir"],
        "poisson_pinn",
    )
    output_dir = run.output_dir

    # One point object is shared by training, metrics, and both algorithms.
    reference_config = configs[algorithms[0]]
    points = build_poisson_points(reference_config)
    torch.manual_seed(int(CONFIG["training"]["seed"]))
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
                pde_weight=float(CONFIG["pinn"]["pde_weight"]),
                boundary_weight=float(CONFIG["pinn"]["boundary_weight"]),
            )["loss"]

        measurement = evaluate_average_sharpness_closure(
            model,
            loss_closure,
            float(evaluation_config["sharpness_scale"]),
            samples=int(evaluation_config["sharpness_samples"]),
            seed=int(evaluation_config["sharpness_seed"]),
            antithetic=bool(evaluation_config.get("antithetic", True)),
            normalized=bool(CONFIG["training"]["perturbation"]["normalized"]),
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
                pde_weight=float(CONFIG["pinn"]["pde_weight"]),
                boundary_weight=float(CONFIG["pinn"]["boundary_weight"]),
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
            normalized=bool(CONFIG["training"]["perturbation"]["normalized"]),
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
    plot_pinn_training_history(results, output_dir / "training_history.png")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "initial_average_sharpness": asdict(initial_sharpness),
                "algorithms": {
                    name: asdict(value) for name, value in metrics.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    run.complete()
    print(f"Plots and metrics written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
